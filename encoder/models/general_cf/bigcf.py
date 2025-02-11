import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
from config.configurator import configs
from models.aug_utils import AdaptiveMask
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class BIGCF(BaseModel):
    def __init__(self, data_handler):
        super(BIGCF, self).__init__(data_handler)

        # prepare adjacency matrix for DCCF
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)),
                                  shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        self.all_h_list = list(plain_adj.row)
        self.all_t_list = list(plain_adj.col)
        self.A_in_shape = plain_adj.shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor(
            [list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))],
            dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()
        self.adaptive_masker = AdaptiveMask(head_list=self.all_h_list, tail_list=self.all_t_list,
                                            matrix_shape=self.A_in_shape)

        # hyper parameters
        self.layer_num = self.hyper_config['layer_num']
        self.intent_num = configs['model']['intent_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.temperature = self.hyper_config['temperature']
        self.cen_weight = self.hyper_config['cen_weight']

        # model parameters
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)

        # train/test
        self.is_training = True
        self.final_embeds = None

        self._init_weight()

    def _init_weight(self):
        init(self.user_embedding.weight)
        init(self.item_embedding.weight)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None

        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        for i in range(0, self.layer_num):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])
            all_embeddings.append(gnn_layer_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        # Bilateral Intent-guided
        u_embeddings, i_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num], 0)
        u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
        i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T

        int_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

        # reparameterization
        noise = torch.randn_like(all_embeddings)
        all_embeddings = all_embeddings + int_embeddings * noise

        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.user_num, self.item_num], 0)


        return self.ua_embedding, self.ia_embedding, all_embeddings, int_embeddings


    def cal_cl_loss(self, users, items, gnn_emb, int_emb):
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temperature)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temperature), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        u_gnn_embs, i_gnn_embs = torch.split(gnn_emb, [self.user_num, self.item_num], 0)
        u_int_embs, i_int_embs = torch.split(int_emb, [self.user_num, self.item_num], 0)

        u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
        u_int_embs = F.normalize(u_int_embs[users], dim=1)

        i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
        i_int_embs = F.normalize(i_int_embs[items], dim=1)

        cl_loss += cal_loss(u_gnn_embs, u_gnn_embs)
        cl_loss += cal_loss(i_gnn_embs, i_gnn_embs)
        cl_loss += cal_loss(u_gnn_embs, i_gnn_embs)

        cl_loss += cal_loss(u_int_embs, u_int_embs)
        cl_loss += cal_loss(i_int_embs, i_int_embs)
        return cl_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, gnn_embeddings, int_embeddings = self.forward()
        ancs, poss, negs = batch_data

        # ELBO reconstruction loss
        u_embeddings = self.ua_embedding[ancs]
        pos_embeddings = self.ia_embedding[poss]
        neg_embeddings = self.ia_embedding[negs]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = self.reg_weight * reg_params(self)

        # collective intents
        cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_weight * cen_loss

        # cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, negs, gnn_embeds, int_embeds)
        cl_loss = self.cl_weight * self.cal_cl_loss(ancs, poss, gnn_embeddings, int_embeddings)
        loss = mf_loss + reg_loss + cl_loss + cen_loss
        losses = {'mf_loss': mf_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'cen_loss': cen_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

    # def full_predict(self, batch_data):
    #     ancs, poss = batch_data
    #     u_embeddings = self.ua_embedding[ancs]
    #     i_embeddings = self.ia_embedding
    #     batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
    #     return batch_ratings
