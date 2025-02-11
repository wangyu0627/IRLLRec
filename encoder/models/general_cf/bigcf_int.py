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


class BIGCF_int(BaseModel):
    def __init__(self, data_handler):
        super(BIGCF_int, self).__init__(data_handler)

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

        # hyper-parameter
        self.intent_num = configs['model']['intent_num']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_int_weight = self.hyper_config['kd_int_weight']
        self.kd_int_temperature = self.hyper_config['kd_int_temperature']
        self.cen_weight = self.hyper_config['cen_weight']

        # model parameters
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.itm_head = nn.Linear(self.embedding_size, 2)

        # train/test
        self.is_training = True
        self.final_embeds = None

        # side information
        # self.usrprf_embeds = configs['usrprf_embeds'].clone().detach().float().cuda()
        # self.itmprf_embeds = configs['itmprf_embeds'].clone().detach().float().cuda()
        self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        # intent information
        self.usrint_embeds = torch.tensor(configs['usrint_embeds']).float().cuda()
        self.itmint_embeds = torch.tensor(configs['itmint_embeds']).float().cuda()
        self.int_mlp = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self.itm_mlp = nn.Linear(2 * self.embedding_size, self.embedding_size)

        self._init_weight()


    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.int_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        init(self.user_embedding.weight)
        init(self.item_embedding.weight)

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list,
            col=self.all_t_list,
            value=edge_alpha,
            sparse_sizes=self.A_in_shape
        ).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None

        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        iaa_embeds = []
        for i in range(0, self.layer_num):
            # Graph-based Message Passing
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])

            all_embeddings.append(gnn_layer_embeddings)
            # Intent-aware Information Aggregation
            u_embeds, i_embeds = torch.split(all_embeddings[i], [self.user_num, self.item_num], 0)
            u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)

            # Intent-aware Augmentation
            head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
            tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
            inten_indices, inten_values = self._adaptive_mask(head_embeds, tail_embeds)
            iaa_layer_embeds = torch_sparse.spmm(inten_indices, inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[-1])
            iaa_embeds.append(iaa_layer_embeds)

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


        return self.ua_embedding, self.ia_embedding, all_embeddings, int_embeddings, iaa_embeds


    def cal_cl_loss(self, users, items, gnn_emb, int_emb):
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.cl_temperature)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.cl_temperature), axis=1)
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

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds


    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, gnn_embeddings, int_embeddings, iaa_embeds = self.forward()
        ancs, poss, negs = batch_data

        # ELBO reconstruction loss
        u_embeddings = self.ua_embedding[ancs]
        pos_embeddings = self.ia_embedding[poss]
        neg_embeddings = self.ia_embedding[negs]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        # mf_loss = self.mf_weight * mf_loss
        # bpr_loss = cal_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings) / u_embeddings.shape[0]

        # bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        # collective intents
        cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_weight * cen_loss

        # cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, negs, gnn_embeds, int_embeds)
        cl_loss = self.cl_weight * self.cal_cl_loss(ancs, poss, gnn_embeddings, int_embeddings)

        # kd_loss
        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)
        kd_loss = cal_infonce_loss(u_embeddings, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeddings, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeddings, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= u_embeddings.shape[0]
        kd_loss *= self.kd_weight

        # kd_int_loss
        int_embeds = torch.mean(torch.stack(iaa_embeds), dim=0)
        user_int_embeds, item_int_embeds = torch.split(int_embeds, [self.user_num, self.item_num], 0)
        anc_int_embeds, pos_int_embeds, neg_int_embeds = self._pick_embeds(user_int_embeds, item_int_embeds, batch_data)
        usrint_embeds = self.int_mlp(self.usrint_embeds)
        itmint_embeds = self.int_mlp(self.itmint_embeds)
        ancint_embeds, posint_embeds, negint_embeds = self._pick_embeds(usrint_embeds, itmint_embeds, batch_data)
        kd_int_loss = cal_infonce_loss(anc_int_embeds, ancint_embeds, usrint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(pos_int_embeds, posint_embeds, posint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(neg_int_embeds, negint_embeds, negint_embeds, self.kd_int_temperature)
        kd_int_loss /= anc_int_embeds.shape[0]
        kd_int_loss *= self.kd_int_weight

        # itm_loss
        # loss_itm_user = self.cal_itm_loss(user_int_embeds, usrint_embeds)
        # loss_itm_item = self.cal_itm_loss(item_int_embeds, itmint_embeds)
        # itm_loss = 0.01*(loss_itm_user + loss_itm_item)

        loss = mf_loss + reg_loss + cl_loss + kd_loss + kd_int_loss + cen_loss #+ itm_loss , 'itm_loss': itm_loss
        losses = {'mf_loss': mf_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss, 'cl_loss': cl_loss, 'kd_int_loss': kd_int_loss, 'cen_loss': cen_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
