import torch as t
import torch
import torch.nn.functional as F

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))

def reg_pick_embeds(embeds_list):
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss

def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()

def ssl_clip_loss(x, y, logit_scale):
    x_norm = x / x.norm(dim=1, keepdim=True)  # [n, d]
    y_norm = y / y.norm(dim=1, keepdim=True)
    logits_t = logit_scale *  x_norm @ y_norm.t()
    logits_g = logits_t.t()
    labels = t.arange(x_norm.shape[0]).cuda()
    loss_t2g = F.cross_entropy(logits_t, labels)
    loss_g2t = F.cross_entropy(logits_g.T, labels)
    loss = (loss_t2g + loss_g2t) / 2

    return loss

def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

def cal_itm_loss(self, x, y):
    concat_pos_embeds = torch.cat((x, y), dim=-1)
    # output_pos = self.itm_mlp(concat_pos_embeds)

    with torch.no_grad():
        x_norm = x / x.norm(dim=1, keepdim=True)  # [n, d]
        y_norm = y / y.norm(dim=1, keepdim=True)
        sim_i2t = self.logit_scale * x_norm @ y_norm.t()
        sim_t2i = sim_i2t.t()

        bs = x.shape[0]
            # Compute softmax weights for interaction-to-text and text-to-interaction similarities
        weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
        weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
        # Avoid self-matching by filling diagonal with zero
        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

    # Negative images for texts
    interaction_embeds_neg = []
    neg_idx = torch.multinomial(weights_t2i, 1).squeeze(1)
    interaction_embeds_neg.append(x[neg_idx])
    interaction_embeds_neg = torch.stack(interaction_embeds_neg, dim=0)

    # Negative texts for interactions
    text_embeds_neg = []
    neg_idx = torch.multinomial(weights_i2t, 1).squeeze(1)
    text_embeds_neg.append(y[neg_idx])
    text_embeds_neg = torch.stack(text_embeds_neg, dim=0)

    # Combine positive and negative samples
    interaction_embeds_all = torch.cat([interaction_embeds_neg, x], dim=0)
    text_embeds_all = torch.cat([y, text_embeds_neg], dim=0)
    concat_neg_embeds = torch.cat((interaction_embeds_all, text_embeds_all), dim=-1)

    # Apply the MLP to both positive and negative samples
    output_pos = self.itm_mlp(concat_pos_embeds)
    output_neg = self.itm_mlp(concat_neg_embeds)

    vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
    vl_output = self.itm_head(vl_embeddings)
    itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).cuda()
    loss_itm = F.cross_entropy(vl_output, itm_labels)

    return loss_itm