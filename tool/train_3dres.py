import argparse
import datetime
import gorilla
import os
import os.path as osp
import shutil
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from gres_model.dataset import build_dataloader, build_dataset
from gres_model.model import MODEL
from gres_model.utils import AverageMeter, get_root_logger


def _print_results_acc(iou_25, iou_50, logger):
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
    logger.info(f"{'=' * 100}")

def get_args():
    parser = argparse.ArgumentParser('SPFormer')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    parser.add_argument('--skip_training', action='store_true', help='skip training')
    parser.add_argument('--dist', action='store_true', help='if distributed')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_machines', type=int, default=1, help='number of machines')
    parser.add_argument('--machine_rank', type=int, default=0, help='rank of machine')
    parser.add_argument('--dist_url', type=str, default="auto", help='distributed training url')
    parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+', help='ids of gpus to use')

    args = parser.parse_args()
    return args

def list_avg(x):
    return sum(x) / len(x)
def dict_val(metric_dict):
    out = {}
    for k, v in metric_dict.items():
        out[k] = v.val
    return out

def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()

    if dataloader.sampler is not None and cfg.dist:
        dataloader.sampler.set_epoch(epoch)

    for i, batch in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)
        # forward
        # torch.autograd.set_detect_anomaly(True)
        loss, log_vars = model(batch, mode='loss')
        # reduce log_vars for multi-gpu
        log_vars = gorilla.reduce_dict(log_vars, average=True)
        if gorilla.is_main_process():
            # meter_dict
            for k, v in log_vars.items():
                if k not in meter_dict.keys():
                    meter_dict[k] = AverageMeter()
                meter_dict[k].update(v.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        
        # clip grad
        if cfg.train.grad_clip > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            
        optimizer.step()

        # time and print
        remain_iter = len(dataloader) * (cfg.train.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']
        
        if i % 10 == 0 and gorilla.is_main_process():
            log_str = f'Epoch [{epoch}/{cfg.train.epochs}][{i}/{len(dataloader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, '
            log_str += f'data_time: {data_time.avg:.2f}, iter_time: {iter_time.avg:.2f}'
            
            meter_dict_print = meter_dict
            for k, v in meter_dict_print.items():
                log_str += f', {k}: {v.val:.4f}'
                writer.add_scalar(f'train/{k}', v.val, i+(epoch-1)*len(dataloader))
            if cfg.train.grad_clip > 0:
                writer.add_scalar('train/grad_total_norm', grad_total_norm, i+(epoch-1)*len(dataloader))
                log_str += f', grad_total_norm: {grad_total_norm:.4f}'
            logger.info(log_str)
            writer.add_scalar('train/learning_rate', lr, i+(epoch-1)*len(dataloader))

    
    gorilla.synchronize()            
    # update lr
    lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    # save checkpoint
    if gorilla.is_main_process():
        save_file = osp.join(cfg.work_dir, 'last.pth')
        meta = dict(epoch=epoch)
        gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)


@torch.no_grad()
def eval(epoch, best_iou, model, dataloader, cfg, logger, writer, save_ckp=False):
    if gorilla.is_main_process():
        logger.info('Validation')
        pious, spious, sp_r_ious, p_r_ious, nt_labels, meta_datas, scan_ids = [], [], [], [], [], [], []
        progress_bar = tqdm(total=len(dataloader))

    model.eval()
    for batch in dataloader:
        result = model(batch, mode='predict')
        piou = result['piou']
        spiou = result['spiou']
        # nt_label = result['nt_label']
        scan_id = result['scan_id']
        if gorilla.is_main_process():
            pious.extend(piou)
            spious.extend(spiou)
            # nt_labels.extend(nt_label)
            scan_ids.extend(scan_id)
            if 'sp_r_iou' in result.keys():
                sp_r_ious.extend(result['sp_r_iou'])
                p_r_ious.extend(result['p_r_iou'])
            if 'meta_datas' in result.keys():
                meta_datas.extend(result['meta_datas'])
            # else:
            #     print('No meta_datas')
            progress_bar.update()
    
    # evaluate
    miou = None
    if gorilla.is_main_process():
        progress_bar.close()
        logger.info('Evaluate referring segmentation: '+str(len(pious)))
        # eval_dict = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
        # eval_type_mask = np.empty(len(scan_ids))
        # for idx, scan_id in enumerate(scan_ids):
        #     eval_type_mask[idx] = eval_dict[meta_datas[idx]['eval_type']]
        #     if nt_labels[idx]:
        #         pious[idx] = torch.tensor(0.0)
        #     if meta_datas[idx]['eval_type'] in ("zt_wo_d", "zt_w_d"):
        #         if nt_labels[idx]:
        #             pious[idx] = torch.tensor(1.0)
        #             spious[idx] = torch.tensor(1.0)
        #         else:
        #             pious[idx] = torch.tensor(0.0)
        #             spious[idx] = torch.tensor(0.0)
                    
        pious = torch.stack(pious, dim=0).cpu().numpy()
        # acc_half_results = {}
        # acc_quarter_results = {}
        # for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
        #     selected_indices = eval_type_mask == eval_dict[sub_group]
        #     selected = pious[selected_indices]      
        #     acc_half_results[sub_group] = (selected > 0.5).sum().astype(float) / selected.size
        #     acc_quarter_results[sub_group] = (selected > 0.25).sum().astype(float) / selected.size
        #     writer.add_scalar('val/'+ sub_group + '_25', acc_quarter_results[sub_group], epoch)
        #     writer.add_scalar('val/'+ sub_group + '_50', acc_half_results[sub_group], epoch)
        
        precision_half = (pious > 0.5).sum().astype(float) / pious.size
        precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
        miou = pious.mean()

        spious = torch.stack(spious, dim=0).cpu().numpy()
        spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
        spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
        spmiou = spious.mean()

        writer.add_scalar('val/mIOU', miou, epoch)
        writer.add_scalar('val/Acc_50', precision_half, epoch)
        writer.add_scalar('val/Acc_25', precision_quarter, epoch)
        writer.add_scalar('val/spmIOU', spmiou, epoch)
        writer.add_scalar('val/spAcc_50', spprecision_half, epoch)
        writer.add_scalar('val/spAcc_25', spprecision_quarter, epoch)
        
        # acc_half_results["overall"] = precision_half
        # acc_quarter_results["overall"] = precision_quarter
        
        logger.info(f'mIoU : {miou}')
        # _print_results_acc(acc_quarter_results, acc_half_results, logger)
        logger.info('mIOU: {:.3f}. Acc_50: {:.3f}. Acc_25: {:.3f}'.format(miou, precision_half,
                                                                    precision_quarter))
        logger.info('spmIOU: {:.3f}. spAcc_50: {:.3f}. spAcc_25: {:.3f}'.format(spmiou, spprecision_half,
                                                                      spprecision_quarter))
        
        if len(sp_r_ious) > 0:
            sp_r_ious = torch.stack(sp_r_ious, dim=0).cpu().numpy()
            p_r_ious = torch.stack(p_r_ious, dim=0).cpu().numpy()
            sp_rmiou = sp_r_ious.mean()
            p_rmiou = p_r_ious.mean()
            writer.add_scalar('val/sp_r_mIOU', sp_rmiou, epoch)
            writer.add_scalar('val/p_r_mIOU', p_rmiou, epoch)
            logger.info('sp_r_mIOU: {:.3f}'.format(sp_rmiou))
            logger.info('p_r_mIOU: {:.3f}'.format(p_rmiou))
    if gorilla.is_main_process() and save_ckp:
        # save
        if miou > best_iou:
            save_file = osp.join(cfg.work_dir, f'best.pth')
            gorilla.save_checkpoint(model, save_file)
        elif epoch > cfg.train.save_epoch:
            save_file = osp.join(cfg.work_dir, f'epoch_{epoch:04d}.pth')
            gorilla.save_checkpoint(model, save_file)

    return miou


def main(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dist = args.dist
    if hasattr(cfg.data.train, 'bidirectional'):
        if cfg.data.train.bidirectional:
            cfg.data.val.bidirectional = True
            cfg.model.decoder.graph_params.num_bond_type *= 2

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./exps', osp.splitext(osp.basename(args.config))[0], timestamp)
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    if gorilla.is_main_process():
        logger = get_root_logger(log_file=log_file)
        logger.info(f'config: {args.config}')
    else:
        logger = None
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # seed
    gorilla.set_random_seed(cfg.train.seed)

    # model
    model = MODEL(**cfg.model).cuda()
    # multi-gpu
    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gorilla.get_local_rank()], find_unused_parameters=True)
    
    count_parameters = gorilla.parameter_count(model)['']
    if gorilla.is_main_process():
        logger.info(f'Parameters: {count_parameters / 1e6:.2f}M')

    # optimizer and scheduler
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # pretrain or resume
    start_epoch = 1
    if args.resume:
        if gorilla.is_main_process(): logger.info(f'Resume from {args.resume}')
        meta = gorilla.resume(model=model, 
                              filename=args.resume, 
                              optimizer=optimizer, 
                              scheduler=lr_scheduler, 
                              strict=False,
                              map_location='cpu')
        start_epoch = meta.get("epoch", 0) + 1
    elif cfg.train.pretrain:
        if gorilla.is_main_process(): logger.info(f'Load pretrain from {cfg.train.pretrain}')
        gorilla.load_checkpoint(model, cfg.train.pretrain, strict=False, map_location='cpu')

    cfg.dataloader.train.batch_size /= args.num_gpus
    cfg.dataloader.train.batch_size = int(cfg.dataloader.train.batch_size)
    if gorilla.is_main_process():
        logger.info(f'Train batch size per gpu: {cfg.dataloader.train.batch_size}')
    # train and val dataset
    if not args.skip_training:
        train_dataset = build_dataset(cfg.data.train, logger)
        train_loader = build_dataloader(train_dataset, dist=args.dist, **cfg.dataloader.train)
    if not args.skip_validate and gorilla.is_main_process():
        val_dataset = build_dataset(cfg.data.val, logger)
        val_loader = build_dataloader(val_dataset, **cfg.dataloader.val)

    # train and val
    if gorilla.is_main_process():
        logger.info('Training')
    best_miou = 0
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if not args.skip_training:
           train(epoch, model, train_loader, optimizer, lr_scheduler, cfg, logger, writer)
        if not args.skip_validate and (epoch % cfg.train.interval == 0) and gorilla.is_main_process():
            miou = eval(epoch, best_miou, model, val_loader, cfg, logger, writer, not args.skip_training)
            if gorilla.is_main_process() and miou > best_miou: 
                best_miou = miou
        writer.flush()
        if args.skip_training:
            break

if __name__ == '__main__':
    args = get_args()
    if args.num_gpus > 1:
        args.dist = True
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)
    gorilla.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,) # use tuple to wrap
    )
    
