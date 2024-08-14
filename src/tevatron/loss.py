import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F


class SimpleContrastiveLoss:
    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = "mean"):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class ListwiseContrastiveLoss:
    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = "mean"):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))

        n_passages = int(y.shape[0] / x.shape[0])

        target = target.to(x.device)

        index = torch.arange(target.size(0), device=target.device)

        teacher_mat = torch.ones(logits.shape, dtype=logits.dtype, device=target.device) * -99

        target = target.view(logits.size(0), -1)

        target = torch.scatter(teacher_mat, dim=-1, index=index.view(logits.size(0), -1),
                               src=target.to(teacher_mat.dtype))

        # sort predicted scores
        logits_sorted, indices_pred = logits.sort(descending=True, dim=-1)

        # sort true w.r.t sorted predicted scores
        true_sorted_by_preds = torch.gather(target, dim=1, index=indices_pred)

        # compute all possible pairs

        excl_in_batch = true_sorted_by_preds != -99
        true_diffs = true_sorted_by_preds[excl_in_batch].view(x.shape[0], n_passages).unsqueeze(
            2) - true_sorted_by_preds.unsqueeze(1)
        pairs_mask = true_diffs > 0

        # inverse rank of passage
        inv_pos_idxs = 1. / torch.arange(1, logits.shape[1] + 1)  # .to(target.device)
        inv_pos_idxs = inv_pos_idxs.to(target.device) * torch.ones(true_sorted_by_preds.shape[0],
                                                                   true_sorted_by_preds.shape[1]).to(target.device)

        weights = torch.abs(
            inv_pos_idxs[excl_in_batch].view(x.shape[0], n_passages).unsqueeze(2) - inv_pos_idxs.unsqueeze(
                1))  # [1, topk, topk]

        scores_diffs = logits_sorted[excl_in_batch].view(x.shape[0], n_passages).unsqueeze(2) - logits_sorted.unsqueeze(
            1)

        # logsumexp trick to avoid inf
        topk = scores_diffs.size(2)
        scores_diffs = scores_diffs.view(1, -1, 1)
        scores_diffs = F.pad(input=-scores_diffs, pad=(0, 1), mode='constant', value=0)
        scores = torch.logsumexp(scores_diffs, 2, True)
        scores = scores.view(-1, n_passages, topk)

        losses = scores * weights  # [bz, topk, topk]
        return torch.mean(losses[pairs_mask])


class ListwisePseudoContrastiveLoss:
    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = "mean"):
        if target is None:
            print("No target provided")
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))

        # labels w.r.t teacher
        target = target.view(logits.size(0), int(logits.size(1) / logits.size(0)))
        target = target.to(logits.device)

        index = torch.arange(target.size(0) * target.size(1), device=logits.device)
        logits = torch.gather(logits, 1, index.view(logits.size(0), -1))

        # sort predicted scores
        logits_sorted, indices_pred = logits.sort(descending=True, dim=-1)

        # sort true w.r.t sorted predicted scores
        true_sorted_by_preds = torch.gather(target, dim=1, index=indices_pred)

        # compute all possible pairs
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        pairs_mask = true_diffs > 0

        # inverse rank of passage
        inv_pos_idxs = 1. / torch.arange(1, logits.shape[1] + 1).to(logits.device)
        weights = torch.abs(inv_pos_idxs.view(1, -1, 1) - inv_pos_idxs.view(1, 1, -1))  # [1, topk, topk]

        # score differences (part of exp)
        scores_diffs = (logits_sorted[:, :, None] - logits_sorted[:, None, :])

        # logsumexp trick to avoid inf
        topk = scores_diffs.size(1)
        scores_diffs = scores_diffs.view(1, -1, 1)
        scores_diffs = F.pad(input=-scores_diffs, pad=(0, 1), mode='constant', value=0)
        scores = torch.logsumexp(scores_diffs, 2, True)
        scores = scores.view(-1, topk, topk)

        losses = scores * weights  # [bz, topk, topk]
        loss = torch.mean(losses[pairs_mask])
        return loss
