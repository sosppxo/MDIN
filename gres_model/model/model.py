import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools

from gres_model.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou, get_iou_prob

from .mdin import MDIN
from transformers import RobertaModel
from pointnet2.pointnet2_utils import FurthestPointSampling

class MODEL(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        sampling_module=None,
        sampling_module_kv=None,
        mdin=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        infer_mode='pos',
        fps_num=512,
        fix_module=[],
        task_type=None,
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.decoder_param = mdin
        self.fps_num = fps_num
        # bert encoder
        self.bert_encoder = RobertaModel.from_pretrained('./backbones/roberta-base')

        self.sampling_module = sampling_module
        # mdin
        self.mdin = MDIN(**mdin, sampling_module=sampling_module, sampling_module_kv=sampling_module_kv, in_channel=media)
        # criterion
        self.criterion = Criterion(**criterion)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.infer_mode = infer_mode
        self.task_type = task_type
        self.init_weights()
        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(MODEL, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, sem2inss, sem_ids, nsubj_inds, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, sem2inss = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss, scenes_len)
        # sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss = self.expand(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss, scenes_len)
        lang_feats = self.bert_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        out = self.mdin(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks)
        
        # if self.sampling_module is None: sp_ref_masks = None
        
        loss, loss_dict = self.criterion(out, gt_pmasks, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels, dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets)

        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, sem2inss, sem_ids, nsubj_inds, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, sem2inss = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss, scenes_len)
        # sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss = self.expand(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss, scenes_len)
        lang_feats = self.bert_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        
        out = self.mdin(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks) 
        ret = self.predict_by_feat(scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, dense_maps)
        if meta_datas[0] is not None: 
            ret['meta_datas'] = meta_datas
            if view_dependents[0] is not None:
                ret['view_dependents'] = view_dependents
        return ret
    
    def predict_by_feat(self, scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, dense_maps):
        # B is 1 when predecit
        spious, pious, sp_r_ious, p_r_ious, pred_pmasks, scan_idss = [], [], [], [], [], []
        nt_labels = []
        b = len(object_idss)
        for i in range(b):
            gt_pmask = gt_pmasks[i]
            gt_spmask = gt_spmasks[i]
            pred_spmask = out['masks'][i].squeeze()

            pred_indis = out['indis'][i] # [n_q, 2]
            # take the 1 
            if self.task_type == 'gres':
                indicate = pred_indis.argmax(-1) == 1
                top = indicate.nonzero(as_tuple=False).squeeze(-1)
                # nt_label
                is_nt = False
                if len(top)==0:
                    is_nt = True
                    piou = torch.tensor(0.0, device=pred_spmask.device)
                    spiou = torch.tensor(0.0, device=pred_spmask.device)
                    pred_spmask = torch.zeros_like(pred_spmask[0], device=pred_spmask.device)
                    pred_pmask = pred_spmask[superpoints]

                else:
                    top_mask = pred_spmask[top] # [k, n_sp]
                    pred_spmask = top_mask.max(0)[0] # [n_sp,]
                    spiou = get_iou(pred_spmask, gt_spmask)
                    pred_pmask = pred_spmask[superpoints]
                    piou = get_iou(pred_pmask, gt_pmask)
                    if not is_nt:
                        pred_pmask_binay = pred_pmask.sigmoid() > 0.5
                        is_nt = True if pred_pmask_binay.sum() < 50 else False
                nt_labels.append(is_nt)
            elif self.task_type == 'res':
                softmax_indis = F.softmax(pred_indis, dim=-1)
                indicate = softmax_indis[:,1]
                top = indicate.argmax()
                pred_spmask = pred_spmask[top]
                spiou = get_iou(pred_spmask, gt_spmask)
                pred_pmask = pred_spmask[superpoints]
                piou = get_iou(pred_pmask, gt_pmask)

            spious.append(spiou.cpu())
            pious.append(piou.cpu())
            pred_pmasks.append(pred_pmask.sigmoid().cpu())
            scan_idss.append(scan_ids[0])
        gt_pmasks = [gt_pmask.cpu() for gt_pmask in gt_pmasks]
        return dict(scan_id=scan_idss, object_ids=object_idss, ann_id=ann_ids, piou=pious, spiou=spious, gt_pmask=gt_pmasks, pred_pmask=pred_pmasks,
                    sp_r_iou=sp_r_ious, p_r_iou=p_r_ious, nt_label = nt_labels)
    
    '''  
    def predict_by_feat(self, scan_ids, object_ids, ann_ids, out, superpoints, gt_pmasks, gt_spmasks):
        # B is 1 when predecit
        gt_pmask = gt_pmasks[0]
        gt_spmask = gt_spmasks[0]
        pred_spmask = out['masks'][-1].squeeze()
        spiou = get_iou(pred_spmask, gt_spmask)

        pred_pmask = pred_spmask[superpoints]
        piou = get_iou(pred_pmask, gt_pmask)

        return dict(scan_id=scan_ids[0], object_id=object_ids[0], ann_id=ann_ids[0], piou=piou, spiou=spiou, gt_pmask=gt_pmask, pred_pmask=pred_pmask)
    '''

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
    
    def expand(self, sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss, scenes_len):
        if scenes_len==None: return sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss
        
        if sp_ins_labels: sp_ins_labels_expand, sem2inss_expand = [], []
        else : sp_ins_labels_expand, sem2inss_expand = None, None
        
        batch_offsets_expand = batch_offsets[0:1]
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            if i==0:
                sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
            else:
                sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                if sp_ins_labels:
                    sp_ins_labels_expand.append(sp_ins_labels[i])
                    # sem2inss_expand.append(sem2inss[i])
        return sp_feats_expand, sp_coords_float_expand, batch_offsets_expand, sp_ins_labels_expand, sem2inss_expand
    
    def expand_and_fps(self, sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss, scenes_len):
        if scenes_len==None: return sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, sem2inss
        
        if sp_ins_labels: sp_ins_labels_expand, sem2inss_expand = [], []
        else : sp_ins_labels_expand, sem2inss_expand = None, None
        
        batch_offsets_expand = batch_offsets[0:1]
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            if i==0:
                sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)
            else:
                sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = torch.cat((fps_seed_sp_expand, fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)),dim=0)
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                if sp_ins_labels:
                    sp_ins_labels_expand.append(sp_ins_labels[i])
                    # sem2inss_expand.append(sem2inss[i])
        return sp_feats_expand, sp_coords_float_expand, fps_seed_sp_expand, batch_offsets_expand, sp_ins_labels_expand, sem2inss_expand