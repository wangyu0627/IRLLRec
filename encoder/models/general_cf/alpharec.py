import torch
import pickle
import numpy as np
# import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from config.configurator import configs
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class AlphaRec(BaseModel):
    def __init__(self, data_handler):
        super(AlphaRec, self).__init__(data_handler)

        self.adj = data_handler.torch_adj
        self.uiadj = data_handler.trn_mat
        self.keep_rate = configs['model']['keep_rate']
        self.lm_model = configs['model']['lm_model']
        self.tau = configs['model']['tau']
        self.edge_dropper = SpAdjEdgeDrop()
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']

        self.init_item_cf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()

        self.init_user_cf_embeds = self.group_agg()
        self.init_embed_shape = self.init_user_cf_embeds.shape[1]

        # self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).cuda(self.device)
        # self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).cuda(self.device)

        multiplier_dict = {
            'bert': 8,
            'roberta': 8,
            'text-embedding-ada-002': 1 / 2,
            'v3': 1 / 2,
            'v3_shuffle': 1 / 2,
        }
        if (self.lm_model in multiplier_dict):
            multiplier = multiplier_dict[self.lm_model]
        else:
            multiplier = 9 / 32

        self.mlp = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embedding_size)
        )

    def group_agg(self):
        """
        Aggregate item embeddings for each user based on interactions, compute the mean.
        Returns:
            Tensor of shape (num_users, embed_dim), user embeddings
        """
        # item embeddings for interactions
        adj_dense = torch.tensor(self.uiadj.toarray()).float().cuda()
        user_item_embeds = torch.matmul(adj_dense, self.init_item_cf_embeds)

        # Count the number of interactions per user to compute the mean
        interaction_counts = adj_dense.sum(dim=1, keepdim=True)
        interaction_counts = torch.clamp(interaction_counts, min=1)  # Avoid division by zero

        # Compute the mean of the item embeddings for each user
        user_embeds = user_item_embeds / interaction_counts

        return user_embeds

    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def forward(self, adj=None, keep_rate=1.0):
        users_cf_emb = self.mlp(self.init_user_cf_embeds)
        items_cf_emb = self.mlp(self.init_item_cf_embeds)

        users_emb = users_cf_emb
        items_emb = items_cf_emb

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)

        for layer in range(self.layer_num):
            all_emb = self._propagate(adj, all_emb)
            embs.append(all_emb)
        # embeds = sum(embeds_list)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.user_num, self.item_num])

        return users, items

    def cal_loss(self, batch_data):
        self.is_training = True

        all_users, all_items = self.forward(self.adj, self.keep_rate)

        ancs, poss, negs = batch_data
        users_emb = all_users[ancs]
        pos_emb = all_items[poss]
        neg_emb = all_items[negs]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim=1)

        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        reg_loss = self.reg_weight * reg_params(self)
        loss = ssm_loss + reg_loss
        losses = {'ssm_loss': ssm_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds