import pickle
import torch as t
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch_sparse
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.general_cf.moe import MoE
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN_int(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_int, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.intent_num = configs['model']['intent_num']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.user_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # prepare for intent
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix(
            (np.ones(len(new_rows)), (new_rows, new_cols)),
            shape=[self.user_num + self.item_num, self.user_num + self.item_num]
        ).tocsr().tocoo()
        self.all_h_list = t.LongTensor(list(plain_adj.row)).cuda()
        self.all_t_list = t.LongTensor(list(plain_adj.col)).cuda()
        self.A_in_shape = plain_adj.shape

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_int_weight = self.hyper_config['kd_int_weight']
        self.kd_int_temperature = self.hyper_config['kd_int_temperature']

        # semantic-embeddings
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        # self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        # self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        # intent information
        self.usrint_embeds = t.tensor(configs['usrint_embeds']).float().cuda()
        self.itmint_embeds = t.tensor(configs['itmint_embeds']).float().cuda()
        self.int_mlp = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        # self.int_mlp = MoE(
        #     input_size=self.usrint_embeds.shape[1],
        #     output_size=self.embedding_size,
        #     num_experts=4,  # Number of experts
        #     top_k=2,  # Number of top experts to use
        #     dropout=0.2,
        #     noise=True
        # )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.int_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = t.nn.functional.normalize(head_embeddings)
        tail_embeddings = t.nn.functional.normalize(tail_embeddings)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list,
            col=self.all_t_list,
            value=edge_alpha,
            sparse_sizes=self.A_in_shape
        ).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = t.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        iaa_embeds = []
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])

            # Intent-aware Information Aggregation
            u_embeds, i_embeds = t.split(embeds_list [i], [self.user_num, self.item_num], 0)
            u_int_embeds = t.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = t.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = t.concat([u_int_embeds, i_int_embeds], dim=0)

            # Intent-aware Augmentation
            head_embeds = t.index_select(int_layer_embeds, 0, self.all_h_list)
            tail_embeds = t.index_select(int_layer_embeds, 0, self.all_t_list)
            inten_indices, inten_values = self._adaptive_mask(head_embeds, tail_embeds)
            iaa_layer_embeds = torch_sparse.spmm(inten_indices, inten_values, self.A_in_shape[0], self.A_in_shape[1], embeds_list[-1])
            iaa_embeds.append(iaa_layer_embeds)

            embeds_list.append(embeds)

        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:], iaa_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, iaa_embeds = self.forward(self.adj, self.keep_rate)

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        # kd_loss
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # kd_int_loss
        int_embeds = t.mean(t.stack(iaa_embeds), dim=0)
        user_int_embeds, item_int_embeds = t.split(int_embeds, [self.user_num, self.item_num], 0)
        anc_int_embeds, pos_int_embeds, neg_int_embeds = self._pick_embeds(user_int_embeds, item_int_embeds, batch_data)
        usrint_embeds = self.int_mlp(self.usrint_embeds)
        itmint_embeds = self.int_mlp(self.itmint_embeds)
        ancint_embeds, posint_embeds, negint_embeds = self._pick_embeds(usrint_embeds, itmint_embeds, batch_data)
        kd_int_loss = cal_infonce_loss(anc_int_embeds, ancint_embeds, usrint_embeds, self.kd_int_temperature) + \
                              cal_infonce_loss(pos_int_embeds, posint_embeds, posint_embeds, self.kd_int_temperature) + \
                              cal_infonce_loss( neg_int_embeds, negint_embeds, negint_embeds, self.kd_int_temperature)
        kd_int_loss /= anc_embeds.shape[0]
        kd_int_loss *= self.kd_int_weight

        loss = bpr_loss + reg_loss + kd_loss + kd_int_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss, 'kd_int_loss': kd_int_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds