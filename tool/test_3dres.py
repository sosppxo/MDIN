import argparse
import gorilla, os
import os.path as osp
import torch
from tqdm import tqdm
import numpy as np
from gres_model.dataset import build_dataloader, build_dataset

from gres_model.model import MODEL
from gres_model.utils.mask_encoder import rle_decode, rle_encode
from gres_model.utils import get_root_logger, save_pred_instances
import json

def get_mask(spmask, superpoint):
    mask = spmask[superpoint]
    return mask

def decode_stimulus_string(s):
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', default=None, type=str, help='directory for output results')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_id', type=int, default=[0], nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_id, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger(log_file=args.checkpoint.replace('.pth', '.log'))

    model = MODEL(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.val, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)

    scan_ids, object_ids, ann_ids, pious, spious, gt_pmasks, pred_pmasks = [], [], [], [], [], [], []
    meta_datas, view_dependents= [], []
    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            res = model(batch, mode='predict')  
    
            scan_ids.extend(res['scan_id'])
            object_ids.extend(res['object_ids'])
            ann_ids.extend(res['ann_id'])
            pious.extend(res['piou'])
            spious.extend(res['spiou'])
            pred_pmasks.extend(
                [
                    rle_encode((pred_pmask>0.5).int().numpy())
                    for pred_pmask in res['pred_pmask']
                ]
            )
            if 'meta_datas' in res:
                meta_datas.extend(res['meta_datas'])
                view_dependents.extend(res['view_dependents'])
            # else:
            #     print('No meta_datas')
            progress_bar.update()
            
        progress_bar.close()

    if len(meta_datas)>0:
        hardness = [decode_stimulus_string(meta_data)[2] for meta_data in meta_datas]
        ious_vd, ious_vind = [], []
        ious_easy, ious_hard = [], []
        for idx, scan_id in enumerate(scan_ids):
            piou = pious[idx]      
            if len(meta_datas)>0:
                if hardness[idx] > 2:
                    ious_hard.append(piou.item())
                else:
                    ious_easy.append(piou.item())
                
                if view_dependents[idx]:
                    ious_vd.append(piou.item())
                else:
                    ious_vind.append(piou.item())   
       
    pious = torch.stack(pious, dim=0).cpu().numpy()
    precision_half = (pious > 0.5).sum().astype(float) / pious.size
    precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
    pmiou = pious.mean()
    # superpoint-level metrics
    spious = torch.stack(spious, dim=0).cpu().numpy()
    spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
    spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
    spmiou = spious.mean()
    logger.info(f'mIoU : {pmiou}')
    logger.info('mIOU: {:.3f}. Acc_50: {:.3f}. Acc_25: {:.3f}'.format(pmiou, precision_half,
                                                                    precision_quarter))
    logger.info('spmIOU: {:.3f}. spAcc_50: {:.3f}. spAcc_25: {:.3f}'.format(spmiou, spprecision_half,
                                                                      spprecision_quarter))        
    if len(meta_datas)>0:
        # vd and vid
        vd_half = (np.array(ious_vd) > 0.5).sum().astype(float) / len(ious_vd)
        vd_quarter = (np.array(ious_vd) > 0.25).sum().astype(float) / len(ious_vd)
        vd_miou = np.array(ious_vd).mean()
        vid_half = (np.array(ious_vind) > 0.5).sum().astype(float) / len(ious_vind)
        vid_quarter = (np.array(ious_vind) > 0.25).sum().astype(float) / len(ious_vind)
        vid_miou = np.array(ious_vind).mean()
        logger.info('vd_Acc@25: {:.4f}. vd_Acc@50: {:.4f}. vd_mIOU: {:.4f}.'.format(vd_quarter, vd_half, vd_miou))
        logger.info('vid_Acc@25: {:.4f}. vid_Acc@50: {:.4f}. vid_mIOU: {:.4f}.'.format(vid_quarter, vid_half, vid_miou))
        # easy and hard
        easy_half = (np.array(ious_easy) > 0.5).sum().astype(float) / len(ious_easy)
        easy_quarter = (np.array(ious_easy) > 0.25).sum().astype(float) / len(ious_easy)
        easy_miou = np.array(ious_easy).mean()
        hard_half = (np.array(ious_hard) > 0.5).sum().astype(float) / len(ious_hard)
        hard_quarter = (np.array(ious_hard) > 0.25).sum().astype(float) / len(ious_hard)
        hard_miou = np.array(ious_hard).mean()
        logger.info('easy_Acc@25: {:.4f}. easy_Acc@50: {:.4f}. easy_mIOU: {:.4f}.'.format(easy_quarter, easy_half, easy_miou))
        logger.info('hard_Acc@25: {:.4f}. hard_Acc@50: {:.4f}. hard_mIOU: {:.4f}.'.format(hard_quarter, hard_half, hard_miou))
        # overall
        half = (pious > 0.5).sum().astype(float) / pious.size
        quarter = (pious > 0.25).sum().astype(float) / pious.size
        miou = pious.mean()
        logger.info('Acc@25: {:.4f}. Acc@50: {:.4f}. mIOU: {:.4f}.'.format(quarter, half, miou))
    else:
        with open(os.path.join(cfg.data.val.data_root,"lookup.json"),'r') as load_f:
            # unique为1, multi为0
            unique_multi_lookup = json.load(load_f)
        unique, multi = [], []
        for idx, scan_id in enumerate(scan_ids):
            if unique_multi_lookup[scan_id][str(object_ids[idx][0])][str(ann_ids[idx])] == 0:
                unique.append(pious[idx])
            else:
                multi.append(pious[idx])
        unique = np.array(unique)
        multi = np.array(multi)
        for u in [0.25, 0.5]:
            logger.info(f'Acc@{u}: \tunique: '+str(round((unique>u).mean(), 4))+' \tmulti: '+str(round((multi>u).mean(), 4))+' \tall: '+str(round((pious>u).mean(), 4)))
        logger.info('mIoU:\t \tunique: '+str(round(unique.mean(), 4))+' \tmulti: '+str(round(multi.mean(), 4))+' \tall: '+str(round(pious.mean(), 4)))
        
    # save output
    if args.out is None:
        output = input('If you want to save the results? (y/n)')
        if output == 'y':
            args.out = os.path.join(os.path.dirname(args.checkpoint), 'results')
        else:
            logger.info('Not saving results.')
            exit()
        
    if args.out:
        logger.info('Saving results...')
        save_pred_instances(args.out, 'pred_instance', scan_ids, object_ids, ann_ids, pred_pmasks)
        logger.info('Done.')

if __name__ == '__main__':
    main()
