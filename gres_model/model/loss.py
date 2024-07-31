import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist
from torch_scatter import scatter_max, scatter_mean
import numpy as np
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight=[1.0, 1.0, 0.5], one_mask=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))
        self.one_mask = one_mask

    @torch.no_grad()
    def forward(self, pred_classes, pred_masks, target, masks):
        '''
        pred_masks: List[Tensor] len(p2c) == B, Tensor.shape == (n, N)
        pred_labels: (B, n_q, 19)
        insts: List[Instances3D]
        '''
        indices = []
        pred_masks_nopadding = []
        for pred_class, pred_mask, tgt, mask in zip(pred_classes, pred_masks, target, masks):
            # remove the padding
            pred_mask = pred_mask.masked_select(mask.unsqueeze(0)).view(pred_mask.shape[0], -1)
            pred_masks_nopadding.append(pred_mask)
            
            tgt_mask = tgt['masks']
            positive_map = tgt['positive_map'].float().to(pred_class.device) # (256)
            if self.one_mask and positive_map.shape[0] > 1:
                positive_map = positive_map[:1]

            if tgt_mask is None:
                indices.append(([], []))
                continue
            
            if self.one_mask:
                tgt_mask = tgt_mask.sum(0, keepdim=True)
            # (n_q, 256) * (256)
            cost_class = -torch.matmul(pred_class, positive_map.transpose(0, 1))

            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_mask + self.cost_weight[1] * cost_dice + self.cost_weight[2] * cost_class)
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return pred_masks_nopadding,\
                [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@torch.jit.script
def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob)**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss

def get_iou(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def get_iou_prob(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    prob in [0, 1]
    '''
    if pad_mask is not None:
        inputs = inputs*pad_mask

    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

@torch.jit.script
def dice_loss_prob(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

@torch.jit.script
def dice_loss_multi_calsses(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float] = None) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from
    https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    input = input.permute(1, 0)
    target = target.permute(1, 0)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1.0 - per_channel_dice

    return loss.mean()

@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fun='bce',
        eos_coef=0.1, temperature=0.07,
        cost_weight=[1.0, 1.0, 0.5],
        match_last_layer=False,
        layer_differ_weight=False,
        one_mask=False,
        pos_loss=True,
    ):
        super().__init__()
        self.loss_fun = loss_fun
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.eos_coef = eos_coef
        self.temperature = temperature
        self.matcher = HungarianMatcher(cost_weight, one_mask)
        self.match_last_layer = match_last_layer
        self.layer_differ_weight = layer_differ_weight
        self.pos_loss = pos_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx



    def loss_masks(self, pred_masks, pred_scores, targets, indices, num_boxes):
        '''Compute bce & dice & score loss for the masks'''
        # score loss
        score_loss = torch.tensor([0.0], device=pred_masks[0].device)
        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_masks[0].device)
        mask_dice_loss = torch.tensor([0.0], device=pred_masks[0].device)
        for mask, score, tgt, (idx_q, idx_gt) in zip(pred_masks, pred_scores, targets, indices):
            if tgt['masks'] is None:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = tgt['masks'][idx_gt]  # (num_inst, N)
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        score_loss = score_loss / num_boxes
        mask_bce_loss = mask_bce_loss / num_boxes
        mask_dice_loss = mask_dice_loss / num_boxes
        # score_loss = score_loss / len(pred_masks)
        # mask_bce_loss = mask_bce_loss / len(pred_masks)
        # mask_dice_loss = mask_dice_loss / len(pred_masks)

        return mask_bce_loss, mask_dice_loss, score_loss


    def loss_indi(self, pred_indis, targets, indices, num_boxes):
        logits = pred_indis.log_softmax(-1) # [B, Q, 2]

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)

        # positive: [0, 1]
        pos_mask = torch.tensor([0, 1], dtype=torch.float, device=logits.device)
        target_mask = torch.zeros_like(logits) # [B, Q, 2]
        target_mask[:, :, 0] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, 0] = 1
        target_sim[src_idx] = pos_mask

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        return loss_ce
    
    #####################################
    # BRIEF dense position-aligned loss #
    #####################################
    def loss_pos_align(self, pred_classes, targets, indices, num_boxes):
        logits = pred_classes.log_softmax(-1) # [B, Q, 256]
        
        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])                  # main object
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])    # attribute(modify)
        pron_positive_map = torch.cat([t["pron_positive_map"] for t in targets])        # pron
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])          # other(auxi)
        rel_positive_map = torch.cat([t["rel_positive_map"] for t in targets])          # relation

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["masks"]) if targets[i]["masks"] is not None else 0
        tgt_idx = torch.cat(tgt_idx)

        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]
        mod_pos = modify_positive_map[tgt_idx]
        pron_pos = pron_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]
        rel_pos = rel_positive_map[tgt_idx]
        # TODO ScanRefer & NR3D
        tgt_weight_pos = tgt_pos * 0.6 + mod_pos * 0.2 + pron_pos * 0.2 + rel_pos*0.1
        # # TODO SR3D (5:1:1:1)/8 = 0.625: 0.125: 0.125: 0.125
        # if outputs["language_dataset"][0] == "sr3d":
        #     tgt_weight_pos = tgt_pos * 0.625 + mod_pos * 0.125 + pron_pos * 0.125 + rel_pos * 0.125

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + pron_pos + rel_pos + other_pos
        target_mask = torch.zeros_like(logits) # [B, Q, 256]
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef

        loss_ce = loss_ce.sum() / num_boxes

        return loss_ce

    ############################
    # BRIEF semantic alignment #
    ############################
    def loss_sem_align(self, proj_tokens, proj_queries, lang_masks, targets, indices, num_boxes):
        # step 1. Contrastive logits
        norm_text_emb = proj_tokens  # B, num_tokens=L, dim=64
        norm_img_emb = proj_queries  # B, num_queries=256, dim=64
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # if query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = lang_masks.sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        pron_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)
        rel_positive_map = torch.zeros(logits.shape, device=logits.device)
        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_pron = torch.cat([
            t['pron_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_rel = torch.cat([
            t['rel_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        pron_positive_map[idx] = pmap_pron
        other_positive_map[idx] = pmap_other
        rel_positive_map[idx] = pmap_rel

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )   # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map>0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map>0), 0)
        pron_positive_logits = -logits.masked_fill(~(pron_positive_map>0), 0)
        rel_positive_logits = -logits.masked_fill(~(rel_positive_map>0), 0)

        pos_modi_term = modi_positive_logits.sum(2)
        pos_pron_term = pron_positive_logits.sum(2)
        pos_rel_term = rel_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map>0).sum(2) + 1e-6
        nb_pron_pos_token = (pron_positive_map>0).sum(2) + 1e-6
        nb_rel_pos_token = (rel_positive_map>0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits+other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token+1e-6) / nb_pos_token
        box_to_token_loss_ = (
            pos_term/nb_pos_token \
            + 0.2*pos_modi_term/nb_modi_pos_token \
            + 0.2*pos_pron_term/nb_pron_pos_token \
            + 0.1*pos_rel_term/nb_rel_pos_token \
            + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map + (modi_positive_map>0) + (pron_positive_map>0) + (rel_positive_map>0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map>0).any(1)] = 0.2
        tmask[(pron_positive_map>0).any(1)] = 0.2
        tmask[(rel_positive_map>0).any(1)] = 0.1
        tmask[torch.arange(len(inds)), inds-1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        pos_pron_term = pron_positive_logits.sum(1)
        pos_rel_term = rel_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term + pos_pron_term + pos_rel_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + pron_positive_map.sum(1) \
             + rel_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj+1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()   

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return tot_loss / num_boxes


    def match_sampled_ins(self, sampled_ins_lbl, obj_ids, sampled_pos, obj_pos):
        '''
        params:
            sampled_ins_lbl: [N_query,]
            obj_ids: list [N_gt]
            sampled_pos: [N_query, 3]
            obj_pos: [N_gt, 3]
        return:
            top1_ids: [N_gt,]
            in_obj_ids: list [N_gt,]
            matched_inds: list[(q_ids, tgt_ids)]
            tgt_counts: [N_gt,]
        '''
        matched_inds = []
        # compute the distance matrix
        dist_mat = torch.cdist(sampled_pos, obj_pos, p=2) # [N_query, N_gt]
        # top1 for each gt
        topk_inds = torch.topk(dist_mat, k=1, dim=0, largest=False, sorted=True)[1].squeeze(0) # [N_gt,]

        matched_inds.extend([(q_id[None], torch.tensor([tgt_id], dtype=torch.int64, device=sampled_pos.device)) for tgt_id, q_id in enumerate(topk_inds)])

        in_obj_ids = []
        for tgt_id, obj_id in enumerate(obj_ids):
            in_obj_id = (sampled_ins_lbl==obj_id).nonzero().squeeze(1) # tensor [N_query,]
            # in_obj_id 去掉topk_inds
            in_obj_id = in_obj_id[~np.isin(in_obj_id.cpu(),topk_inds.cpu())]
            in_obj_ids.append(in_obj_id)
            if in_obj_id.numel() == 0: continue
            else:
                matched_inds.extend([(q_id[None], torch.tensor([tgt_id], dtype=torch.int64, device=sampled_pos.device)) for q_id in in_obj_id])
        
        # matched_inds: [(q_id, tgt_id), ...]
        matched_inds_np = np.array([(matched_inds[i][0][0].cpu(),matched_inds[i][1][0].cpu()) for i in range(len(matched_inds))])
        tgt_ids = matched_inds_np[:, 1]
        _, tgt_counts = np.unique(tgt_ids, return_counts=True)

        matched_inds = torch.from_numpy(matched_inds_np).transpose(0,1).to(sampled_pos.device)
        matched_inds = (matched_inds[0], matched_inds[1])
        return topk_inds, in_obj_ids, matched_inds, tgt_counts

    def get_topk_seed_inds(self, seed_pos, obj_pos):
        '''
        params:
            seed_ins_lbl: [N_seed,]
            obj_ids: list [N_gt]
            seed_pos: [N_seed, 3]
            obj_pos: [N_gt, 3]
        return:
            top1_ids: [N_gt,]
        '''
        # compute the distance matrix
        dist_mat = torch.cdist(seed_pos, obj_pos, p=2) # [N_query, N_gt]
        # top1 for each gt
        topk_inds = torch.topk(dist_mat, k=1, dim=0, largest=False, sorted=True)[1].squeeze(0) # [N_gt,]
        return topk_inds

    def get_layer_loss(self, layer, aux_outputs, pad_masks, target, indices=None, lang_masks=None, proj_tokens=None):
        loss_out = {}

        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        proj_queries = aux_outputs['proj_queries']
        pred_indis = aux_outputs['indis']

        pred_masks_nopadding = []
        for pred_mask, mask in zip(pred_masks, pad_masks):
            pred_masks_nopadding.append(pred_mask.masked_select(mask.unsqueeze(0)).view(pred_mask.shape[0], -1))
        pred_masks = pred_masks_nopadding
        # pred_masks: List[Tensor (n_query, M)]
        # target_masks: List[Tensor (n_tgt, N)] （None if n_tgt==0)

        num_insts = sum(len(inds[1]) for inds in indices)
        num_insts = torch.as_tensor(
            [num_insts], dtype=torch.float,
            device=pred_masks[0].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_insts)

        indi_loss = self.loss_indi(pred_indis, target, indices, num_insts)
        mask_bce_loss, mask_dice_loss, score_loss = self.loss_masks(pred_masks, pred_scores, target, indices, num_insts)
        
        if proj_tokens is not None:
            sem_loss = self.loss_sem_align(proj_tokens, proj_queries, lang_masks, target, indices, num_insts)
            # sem_loss = torch.tensor([0.0], device=pred_masks[0].device)
        else:
            sem_loss = torch.tensor([0.0], device=pred_masks[0].device)
        
        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['sem_loss'] = sem_loss
        loss_out['indi_loss'] = indi_loss

        loss = (
            self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss +
            self.loss_weight[2] * score_loss + self.loss_weight[4] * indi_loss +
            self.loss_weight[5] * sem_loss)

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def forward(self, pred, gt_pmasks, gt_spmasks, sp_ref_masks=None, object_idss=None, sp_ins_labels=None, dense_maps=None, lang_masks=None, fps_seed_sp=None, sp_coords_float=None, batch_offsets=None):
        '''
            pred_masks: List[Tensor (1, M)]
            并行: Tensor (B, 1, M)
            pred_scores: (B, n, 1) or [(B, n, 1)]
            gt_pmasks: List[Tensor (1, N)]
            gt_sp_masks: List[Tensor (M)]
            sp_ref_masks: List[Tensor (M)]
        '''
        loss_out = {}

        pred_scores = pred['scores']
        pred_masks = pred['masks']
        pad_masks = ~pred['batch_mask']
        pred_indis = pred['indis']
        proj_tokens = pred['proj_tokens']
        proj_queries = pred['proj_queries']
        # proj_instances = pred['proj_instances']
        ref_padding = pad_sequence(sp_ref_masks, batch_first=True, padding_value=0)

        ref_scores = pred['ref_scores']
        sample_inds = pred['sample_inds']

        ref_scores_kv = pred['ref_scores_kv']

        pred_masks_nopadding = []
        for pred_mask, mask in zip(pred_masks, pad_masks):
            pred_mask = pred_mask.masked_select(mask.unsqueeze(0)).view(pred_mask.shape[0], -1)
            pred_masks_nopadding.append(pred_mask)
        pred_masks = pred_masks_nopadding

        sp_coords_float = self.get_batches(sp_coords_float, batch_offsets)[0]
        fps_seed_sp = fps_seed_sp.long()
        target = []
        indices = []
        topk_seed_inds = []
        matched_seed_inds = []
        obj_poss = []

        gt_spmasks = pad_sequence(gt_spmasks, batch_first=True, padding_value=0)
        sample_seed_lbl = gt_spmasks.gather(1, fps_seed_sp)
        seed_pos = sp_coords_float.gather(1, fps_seed_sp.unsqueeze(-1).repeat(1, 1, 3))
        
        for b in range(len(object_idss)):
            obj_ids = object_idss[b]
            if len(obj_ids) == 0:
                indices.append(
                    (torch.as_tensor([], dtype=torch.int64, device=pred_scores.device), 
                        torch.as_tensor([], dtype=torch.int64, device=pred_scores.device))
                )
                target.append(
                    {
                        'masks': None,
                        'positive_map': dense_maps[b]['positive_map'][:1].repeat(len(object_idss[b]), 1).to(pred_scores.device),
                        'modify_positive_map': dense_maps[b]['modify_positive_map'][:1].repeat(len(object_idss[b]), 1).to(pred_scores.device),
                        'pron_positive_map': dense_maps[b]['pron_positive_map'][:1].repeat(len(object_idss[b]), 1).to(pred_scores.device),
                        'other_entity_map': dense_maps[b]['other_entity_map'][:1].repeat(len(object_idss[b]), 1).to(pred_scores.device),
                        'rel_positive_map': dense_maps[b]['rel_positive_map'][:1].repeat(len(object_idss[b]), 1).to(pred_scores.device),
                    }
                )
                topk_seed_inds.append(torch.as_tensor([], dtype=torch.int64, device=pred_scores.device))
                matched_seed_inds.append(torch.as_tensor([], dtype=torch.int64, device=pred_scores.device))
                obj_poss.append(torch.as_tensor([], dtype=torch.int64, device=pred_scores.device))
                continue
            # seed label
            seed_ins_lbl = sp_ins_labels[b][fps_seed_sp[b]]
            # query label   
            sampled_ins_lbl = seed_ins_lbl[sample_inds[b]]
            # query pos
            sampled_pos = seed_pos[b][sample_inds[b]]
            # masks
            masks = torch.stack([sp_ins_labels[b]==obj_id for obj_id in obj_ids])
            obj_pos = torch.stack([
                scatter_mean(sp_coords_float[b].masked_select(pad_masks[b].unsqueeze(1)).view(-1,3), 
                                mask.long(), 
                                dim=0)[1]
                for mask in masks
            ])
            obj_poss.append(obj_pos)
            # [N_gt,]
            topk_seed, in_obj_seed, matched_seed, tgt_counts_seed= self.match_sampled_ins(seed_ins_lbl, obj_ids, seed_pos[b], obj_pos)
            # topk_seed = self.get_topk_seed_inds(seed_pos[b], obj_pos)
            # [N_gt,], list [N_gt], (q_ids, tgt_ids)
            topk_inds, in_obj_ids, matched_inds, tgt_counts = self.match_sampled_ins(sampled_ins_lbl, obj_ids, sampled_pos, obj_pos)
            # repeat gt according to tgt_counts
            masks = torch.cat([masks[i, None].repeat(tgt_counts[i],1) for i in range(len(masks))], dim=0)
            target.append(
                {
                    'masks': masks,
                    'positive_map': dense_maps[b]['positive_map'][:1].repeat(len(masks), 1).to(pred_scores.device),
                    'modify_positive_map': dense_maps[b]['modify_positive_map'][:1].repeat(len(masks), 1).to(pred_scores.device),
                    'pron_positive_map': dense_maps[b]['pron_positive_map'][:1].repeat(len(masks), 1).to(pred_scores.device),
                    'other_entity_map': dense_maps[b]['other_entity_map'][:1].repeat(len(masks), 1).to(pred_scores.device),
                    'rel_positive_map': dense_maps[b]['rel_positive_map'][:1].repeat(len(masks), 1).to(pred_scores.device),
                }
            )
            assert len(matched_inds[0]) == len(masks)
            indices.append(matched_inds)
            topk_seed_inds.append(topk_seed)
            matched_seed_inds.append(matched_seed)

        num_insts = sum(len(inds[1]) for inds in indices)
        num_insts = torch.as_tensor(
            [num_insts], dtype=torch.float,
            device=pred_masks[0].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_insts)
        indi_loss = self.loss_indi(pred_indis, target, indices, num_insts)
        mask_bce_loss, mask_dice_loss, score_loss = self.loss_masks(pred_masks, pred_scores, target, indices, num_insts)
        # mask_dice_loss *= 4

        if proj_tokens is not None:
            sem_loss = self.loss_sem_align(proj_tokens, proj_queries, lang_masks, target, indices, num_insts)
            # sem_loss = torch.tensor([0.0], device=pred_masks[0].device)
        else:
            sem_loss = torch.tensor([0.0], device=pred_masks[0].device)
    
        # sample_loss = torch.tensor([0.0], device=pred_masks[0].device)
        # sample loss
        if ref_scores is not None:
            # seed
            ref_padding_seed_ = ref_padding.gather(1, fps_seed_sp)
            ref_padding_seed = ref_padding_seed_ * 0.5
            # [B, M]
            for b in range(len(topk_seed_inds)):
                # ref_padding_seed[b][topk_seed_inds[b]] = 1.0
                # topk_pos = seed_pos[b][topk_seed_inds[b]]
                if len(matched_seed_inds[b]) == 0:
                    continue
                dist_seeds = torch.cdist(seed_pos[b], obj_poss[b], p=2)
                weight = torch.exp(-0.5 * ((dist_seeds) / 1) ** 2)
                for idx, obj in zip(matched_seed_inds[b][0], matched_seed_inds[b][1]):
                    ref_padding_seed[b][idx] = weight[idx][obj]
                ref_padding_seed[b][topk_seed_inds[b]] = 1.0

            if self.loss_fun=='focal':
                sample_criterion = SigmoidFocalClassificationLoss()
                seed_pad_masks = pad_masks.gather(1, fps_seed_sp)
                cls_weights = seed_pad_masks.float()
                cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
                cls_weights /= torch.clamp(cls_normalizer, min=1.0)
                # focal loss
                sample_loss = sample_criterion(ref_scores.unsqueeze(-1), ref_padding_seed.unsqueeze(-1).float(), weights=cls_weights)
                sample_loss = (sample_loss.squeeze(-1)*seed_pad_masks).sum(-1) # / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
                
            elif self.loss_fun=='bce':
                # bce loss
                sample_loss = F.binary_cross_entropy_with_logits(ref_scores, ref_padding_seed.float(), reduction='none')
                sample_loss = (sample_loss*pad_masks).sum(-1)  / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
            else:
                raise NotImplementedError

        # # sample loss
        # if ref_scores_kv is not None:
        #     # [B, M]
        #     ref_scores = pred['ref_scores']

        #     if self.loss_fun=='focal':
        #         sample_criterion_kv = SigmoidFocalClassificationLoss()
        #         cls_weights = pad_masks.float()
        #         cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        #         cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        #         # focal loss
        #         sample_loss_kv = sample_criterion_kv(ref_scores_kv.unsqueeze(-1), ref_padding.unsqueeze(-1).float(), weights=cls_weights)
        #         sample_loss_kv = (sample_loss.squeeze(-1)*pad_masks).sum(-1) # / pad_masks.sum(-1)
        #         sample_loss_kv = sample_loss.mean()
                
        #     elif self.loss_fun=='bce':
        #         # bce loss
        #         sample_loss_kv = F.binary_cross_entropy_with_logits(ref_scores_kv, ref_padding.float(), reduction='none')
        #         sample_loss_kv = (sample_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
        #         sample_loss_kv = sample_loss.mean()
        #     else:
        #         raise NotImplementedError
        # else:
        #     sample_loss_kv = torch.tensor([0.0], device=pred_masks[0].device)
        
        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['sem_loss'] = sem_loss
        loss_out['indi_loss'] = indi_loss
        loss_out['sample_loss'] = sample_loss
        
        loss = (
                self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss + self.loss_weight[3] * sample_loss + self.loss_weight[4] * indi_loss +
                self.loss_weight[5] * sem_loss
                )
        
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, pad_masks, target, indices, lang_masks, proj_tokens)
                if self.layer_differ_weight:
                    # 1/7, 2/7, 3/7, 4/7, 5/7, 6/7
                    loss += loss_i * ((i+1) / (len(pred['aux_outputs']) + 1))
                else:
                    loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss

        return loss, loss_out