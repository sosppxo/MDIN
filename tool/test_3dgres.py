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

def get_query_iou(query_a, query_b):
    mask_a = query_a.sigmoid()
    mask_b = query_b.sigmoid()
    # thresholding
    binarized_a = (query_a >= 0.5).float()
    binarized_b = (query_b >= 0.5).float()
    intersection = (binarized_a * binarized_b).sum(-1)
    union = binarized_b.sum(-1) + binarized_a.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def _print_results_acc(iou_25, iou_50, mious, logger):
    logger.info(f"{'=' * 100}")
    logger.info("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
               .format("IoU", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
    logger.info(f"{'-' * 100}")
    line_1_str = '{:<12}'.format("0.25")
    for sub_group_type, score in iou_25.items():
        line_1_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_1_str)
    line_2_str = '{:<12}'.format("0.50")
    for sub_group_type, score in iou_50.items():
        line_2_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_2_str)
    line_3_str = '{:<12}'.format("miou")
    for sub_group_type, score in mious.items():
        line_3_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_3_str)
    logger.info(f"{'=' * 100}\n")
    # print(f"{'=' * 100}")
    # print("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
    #           .format("IoU", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
    # print(f"{'-' * 100}")
    # line_1_str = '{:<12}'.format("0.25")
    # for sub_group_type, score in iou_25.items():
    #     line_1_str += '{:<12.1f}'.format(score * 100)
    # print(line_1_str)
    # line_2_str = '{:<12}'.format("0.50")
    # for sub_group_type, score in iou_50.items():
    #     line_2_str += '{:<12.1f}'.format(score * 100)
    # print(line_2_str)
    # print(f"{'=' * 100}\n")

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
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)   # 改训练集or验证集

    scan_ids, object_ids, ann_ids, pious, spious, gt_pmasks, pred_pmasks = [], [], [], [], [], [], []
    meta_datas = []
    nt_labels = []
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
            nt_labels.extend(res['nt_label'])

            # if 'meta_datas' in res:
            #     meta_datas.extend(res['meta_datas'])
            #     view_dependents.extend(res['view_dependents'])
            if 'meta_datas' in res:
                meta_datas.extend(res['meta_datas'])
            else:
                print('No meta_datas')

            progress_bar.update()
            
        progress_bar.close()

    
    iou_out = {}
    eval_dict = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
    eval_type_mask = np.empty(len(scan_ids))
    for idx, scan_id in enumerate(scan_ids):
        ann_id = int(ann_ids[idx])
        eval_type = meta_datas[idx]['eval_type']
        eval_type_mask[idx] = eval_dict[eval_type]
        
        if scan_id not in iou_out:
            iou_out[scan_id] = {}
        if ann_id not in iou_out[scan_id]:
            iou_out[scan_id][ann_id] = {}
        
        iou_out[scan_id][ann_id]["eval_type"] = eval_type
        iou_out[scan_id][ann_id]["object"] = object_ids[idx]
        iou_out[scan_id][ann_id]["pred_nt"] = float(nt_labels[idx])
        
        # 如果被预测为 nt，iou = 0
        if nt_labels[idx]:
            iou_out[scan_id][ann_id]["ori_iou"] = float(pious[idx].cpu().numpy())
            pious[idx] = torch.tensor(0.0)
        
        if eval_type in ("zt_wo_d", "zt_w_d"):
            if nt_labels[idx]:
                pious[idx] = torch.tensor(1.0)
            else:
                pious[idx] = torch.tensor(0.0)
        
        iou_out[scan_id][ann_id]["iou"] = float(pious[idx].cpu().numpy())
    
    iou_path = args.checkpoint.replace('.pth', '_outdict.json')
    
    save_inform = 0
    if save_inform == 1:
        with open(iou_path, "w") as json_file:
            json.dump(iou_out, json_file, indent=2)    

       
    pious = torch.stack(pious, dim=0).cpu().numpy()     
    acc_half_results = {}
    acc_quarter_results = {}
    meam_ious = {}
    for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
        selected_indices = eval_type_mask == eval_dict[sub_group]
        selected = pious[selected_indices]
        meam_ious[sub_group] = selected.mean()
        acc_half_results[sub_group] = (selected > 0.5).sum().astype(float) / selected.size
        acc_quarter_results[sub_group] = (selected > 0.25).sum().astype(float) / selected.size


        # if len(meta_datas)>0:
        #     pass
            # if hardness[idx] > 2:
            #     ious_hard.append(piou.item())
            # else:
            #     ious_easy.append(piou.item())
            
            # if view_dependents[idx]:
            #     ious_vd.append(piou.item())
            # else:
            #     ious_vind.append(piou.item())
    
    
    precision_half = (pious > 0.5).sum().astype(float) / pious.size
    precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
    pmiou = pious.mean()
    # superpoint-level metrics
    spious = torch.stack(spious, dim=0).cpu().numpy()
    spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
    spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
    spmiou = spious.mean()
    acc_half_results["overall"] = precision_half
    acc_quarter_results["overall"] = precision_quarter
    meam_ious["overall"] = pmiou
    
    logger.info('============================= Points =============================')
    logger.info(f'Point mIoU : {pmiou}')
    _print_results_acc(acc_quarter_results, acc_half_results, meam_ious, logger)
    #     iou_dict[scan_id+'_'+str(object_id).zfill(3)+'_'+str(ann_id).zfill(3)] = piou.item()
    # iou_path = os.path.join(os.path.dirname(args.checkpoint), 'ious.json')
    # # write to json
    # with open(iou_path, 'w') as f:
    #     json.dump(iou_dict, f)
    
    # logger.info('Evaluate referring segmentation')
    # # point-level metrics
    # pious = torch.stack(pious, axis=0).cpu().numpy()
    # # superpoint-level metrics
    # spious = torch.stack(spious, axis=0).cpu().numpy()
    # spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
    # spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
    # spmiou = spious.mean()
    # logger.info('sp_Acc@25: {:.4f}. sp_Acc@50: {:.4f}. sp_mIOU: {:.4f}.'.format(spprecision_quarter, spprecision_half, spmiou))
    
    # print(pious.size)
    
    # if len(meta_datas)==0:
    #     with open(os.path.join(cfg.data.val.data_root,"lookup.json"),'r') as load_f:
    #         # unique为1, multi为0
    #         unique_multi_lookup = json.load(load_f)
    #     unique, multi = [], []
    #     for idx, scan_id in enumerate(scan_ids):
    #         if unique_multi_lookup[scan_id][str(object_ids[idx])][str(ann_ids[idx])] == 0:
    #             unique.append(pious[idx])
    #         else:
    #             multi.append(pious[idx])
    #     unique = np.array(unique)
    #     multi = np.array(multi)
    #     for u in [0.25, 0.5]:
    #         logger.info(f'Acc@{u}: \tunique: '+str(round((unique>u).mean(), 4))+' \tmulti: '+str(round((multi>u).mean(), 4))+' \tall: '+str(round((pious>u).mean(), 4)))
        # logger.info('mIoU:\t \tunique: '+str(round(unique.mean(), 4))+' \tmulti: '+str(round(multi.mean(), 4))+' \tall: '+str(round(pious.mean(), 4)))
        
    # if len(meta_datas)>0:
    #     # vd and vid
    #     vd_half = (np.array(ious_vd) > 0.5).sum().astype(float) / len(ious_vd)
    #     vd_quarter = (np.array(ious_vd) > 0.25).sum().astype(float) / len(ious_vd)
    #     vd_miou = np.array(ious_vd).mean()
    #     vid_half = (np.array(ious_vind) > 0.5).sum().astype(float) / len(ious_vind)
    #     vid_quarter = (np.array(ious_vind) > 0.25).sum().astype(float) / len(ious_vind)
    #     vid_miou = np.array(ious_vind).mean()
    #     logger.info('vd_Acc@25: {:.4f}. vd_Acc@50: {:.4f}. vd_mIOU: {:.4f}.'.format(vd_quarter, vd_half, vd_miou))
    #     logger.info('vid_Acc@25: {:.4f}. vid_Acc@50: {:.4f}. vid_mIOU: {:.4f}.'.format(vid_quarter, vid_half, vid_miou))
    #     # easy and hard
    #     easy_half = (np.array(ious_easy) > 0.5).sum().astype(float) / len(ious_easy)
    #     easy_quarter = (np.array(ious_easy) > 0.25).sum().astype(float) / len(ious_easy)
    #     easy_miou = np.array(ious_easy).mean()
    #     hard_half = (np.array(ious_hard) > 0.5).sum().astype(float) / len(ious_hard)
    #     hard_quarter = (np.array(ious_hard) > 0.25).sum().astype(float) / len(ious_hard)
    #     hard_miou = np.array(ious_hard).mean()
    #     logger.info('easy_Acc@25: {:.4f}. easy_Acc@50: {:.4f}. easy_mIOU: {:.4f}.'.format(easy_quarter, easy_half, easy_miou))
    #     logger.info('hard_Acc@25: {:.4f}. hard_Acc@50: {:.4f}. hard_mIOU: {:.4f}.'.format(hard_quarter, hard_half, hard_miou))
    #     # overall
    #     half = (pious > 0.5).sum().astype(float) / pious.size
    #     quarter = (pious > 0.25).sum().astype(float) / pious.size
    #     miou = pious.mean()
    #     logger.info('Acc@25: {:.4f}. Acc@50: {:.4f}. mIOU: {:.4f}.'.format(quarter, half, miou))
        
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
