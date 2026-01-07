import torch
import torch.nn.functional as F
from torch import nn

from _utils import gen_triplets


def get_all_triplets_indices2(labels, ref_labels=None):
    # if batch_size = 7
    # A = sames.unsqueeze(2) -> 7 x 7 x 1
    # B = diffs.unsqueeze(1) -> 7 x 1 x 7
    # A * B -> [7 x 7 x 1->repeat->7] * [7 x 1->repeat->7 x 7]
    #           a11 a11..........a11     b11 b12..........b17 : a11=0 -> pass
    #           a12 a12..........a12     b11 b12..........b17 : if a12=1 & b15=1 -> find triplet: 1=anc 2=pos 5=neg
    #           ...
    #           a77...
    # torch.where([7 x 7 x 7]) -> all 1s in the matrix's 3d index: ([x1,...],[y1,...],[z1,...])
    # so xi, yi, zi is index of anchor, positive and negative in the batch
    if ref_labels is None:
        ref_labels = labels

    sames = labels @ ref_labels.T > 0
    diffs = ~sames

    # only for self comparison & xbm
    sames[:, -labels.size(0) :].fill_diagonal_(False)

    # NOTE: gen triplets using AP=1 & AN=0 but lack of PN=0, which will not harm the TripletLoss,
    # because another triplet may use P as A.
    return torch.where(sames.unsqueeze(2) * diffs.unsqueeze(1))


class SCTLoss(nn.Module):
    def __init__(self, method, lam=1, temperature=0.1, margin=0.25):
        super().__init__()

        self.triplet_fnc = self.nca_based_fnc if "nca" in method else self.tri_based_fnc
        self.lam = lam
        self.temperature = temperature
        self.margin = margin

    def _forward(self, batch, labels, ref_batch=None, ref_labels=None, _type="full"):
        # feature normalization
        batch = F.normalize(batch, p=2, dim=1)

        if ref_batch is None:
            ref_batch = batch
        else:
            ref_batch = F.normalize(ref_batch, p=2, dim=1)

        # use negative Similarity Matrix as distance
        sim_mat = batch @ ref_batch.T

        anc_idxes, pos_idxes, neg_idxes = gen_triplets(labels, ref_labels)
        S_ap = sim_mat[anc_idxes, pos_idxes]
        S_an = sim_mat[anc_idxes, neg_idxes]

        hard_conditions = (S_an > S_ap) | (S_an > 1 - self.margin)

        N_hard = hard_conditions.float().sum()

        if N_hard > 0:
            anc_idxes2 = anc_idxes[hard_conditions]
            neg_idxes2 = neg_idxes[hard_conditions]

            S_an2 = sim_mat[anc_idxes2, neg_idxes2]

            loss_hard_triplet = S_an2.mean()
        else:
            loss_hard_triplet = 0

        if _type == "part":
            return loss_hard_triplet, N_hard

        easy_conditions = (S_an < S_ap) & (S_ap - S_an < self.margin) & (S_an < 1 - self.margin)
        N_easy = easy_conditions.float().sum()

        if N_easy > 0:
            anc_idxes2 = anc_idxes[easy_conditions]
            pos_idxes2 = pos_idxes[easy_conditions]
            neg_idxes2 = neg_idxes[easy_conditions]

            S_ap2 = sim_mat[anc_idxes2, pos_idxes2]
            S_an2 = sim_mat[anc_idxes2, neg_idxes2]

            loss_easy_triplet = self.triplet_fnc(S_ap2, S_an2)
        else:
            loss_easy_triplet = 0

        return loss_hard_triplet, N_hard, loss_easy_triplet, N_easy

    def forward(self, batch, labels, xbm):

        loss_hard_triplet, N_hard, loss_easy_triplet, N_easy = self._forward(batch, labels)

        # current = 896
        # while N_hard == 0:
        #     current += 128
        #     if current > xbm.max_size:
        #         break
        #     xbm_feats, xbm_labels = xbm.get()
        #     ref_embeddings = xbm_feats[-current:]
        #     ref_labels = xbm_labels[-current:]
        #     if N_semi == 0:
        #         # print(f"no semi-hard triplets, use full xbm[-{current}:]")
        #         loss_hard_triplet, N_hard, loss_semi_triplet, N_semi = self._forward(
        #             batch, labels, ref_embeddings, ref_labels, "full"
        #         )
        #     else:
        #         # print(f"no hard triplets, use part xbm[-{current}:]")
        #         loss_hard_triplet, N_hard = self._forward(batch, labels, ref_embeddings, ref_labels, "part")

        if N_hard == 0 and xbm is not None:
            xbm_feats, xbm_labels = xbm.get()
            if N_easy == 0:
                # print("no semi-hards, use full xbm")
                loss_hard_triplet, N_hard, loss_easy_triplet, N_easy = self._forward(
                    batch, labels, xbm_feats, xbm_labels, "full"
                )
            else:
                # print("no hards, use part xbm")
                loss_hard_triplet, N_hard = self._forward(batch, labels, xbm_feats, xbm_labels, "part")

        loss = loss_easy_triplet + self.lam * loss_hard_triplet

        N = N_hard + N_easy
        hn_ratio = N_hard / N if N > 0 else 0

        return loss, hn_ratio, N

    def nca_based_fnc(self, S_ap, S_an):
        ap_an_pairs = torch.stack([S_ap, S_an], 1)
        # [:, 0] means only pick the -log(expS_ap/(expS_ap+expS_an)) and ignore the -log(expS_an/(expS_ap+expS_an))
        return -F.log_softmax(ap_an_pairs / self.temperature, dim=1)[:, 0].mean()

    def tri_based_fnc(self, S_ap, S_an):
        violation = S_an - S_ap + self.margin
        return F.relu(violation).mean()


if __name__ == "__main__":
    from _utils import gen_test_data

    e, _, l = gen_test_data(20, 10, 16, True)

    loss = SCTLoss("sct-margin")
    print(loss._forward(e, l))
