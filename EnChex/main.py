# --------------------------------------------------------
# Swin Transformer (Hybrid CNN-Swin + XAI Mod)
# Copyright (c) 2021 Microsoft, extended for hybrid/XAI
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from sklearn.metrics import roc_auc_score

# ---- XAI imports ----
from xai_utils.explainability import GradCAM, visualize_prototypes  # Adjust path if needed

def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script',
        add_help=False,
        conflict_handler='resolve'
    )
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='(IGNORED) mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0, help='local rank for DistributedDataParallel')

    # NIH dataset
    parser.add_argument("--trainset", type=str, required=True, help='path to train dataset')
    parser.add_argument("--validset", type=str, required=True, help='path to validation dataset')
    parser.add_argument("--testset", type=str, required=True, help='path to test dataset')
    parser.add_argument("--train_csv_path", type=str, required=True, help='path to train csv file')
    parser.add_argument("--valid_csv_path", type=str, required=True, help='path to validation csv file')
    parser.add_argument("--test_csv_path", type=str, required=True, help='path to test csv file')
    parser.add_argument("--num_mlp_heads", type=int, default=3, choices=[0, 1, 2, 3], help='number of mlp layers at end of network')

    # ---- XAI options ----
    parser.add_argument('--xai', action='store_true', help='Enable XAI visualizations (GradCAM/Prototypes)')
    parser.add_argument('--xai-vis-freq', type=int, default=10, help='Epochs between XAI visualizations')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def main(config, args):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(config)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # ---- Hybrid/prototype-aware loss ----
    if config.MODEL.TYPE == 'hybrid_swin_cnn':
        def hybrid_loss_fn(outputs, targets):
            cls_loss = F.binary_cross_entropy_with_logits(outputs['logits'], targets)
            proto_loss = outputs.get('prototype_loss', torch.tensor(0.0, device=cls_loss.device))
            return cls_loss + 0.1 * proto_loss
        criterion = hybrid_loss_fn
    elif config.AUG.MIXUP > 0.:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    scaler = GradScaler()  # Native PyTorch AMP

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, scaler)
        acc1, acc5, loss = validate(config, data_loader_val, model, is_validation=True, args=args)
        logger.info(f"Mean Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.2f}%")
        logger.info(f"Mean Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        throughput(data_loader_test, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, scaler)

        # ---- Prototype projection step (every 5 epochs for hybrid) ----
        if config.MODEL.TYPE == 'hybrid_swin_cnn' and epoch % 5 == 0:
            if hasattr(model_without_ddp, 'prototype_layer'):
                logger.info('Projecting prototypes...')
                def feature_extractor(images):
                    model_without_ddp.eval()
                    with torch.no_grad():
                        cnn_features = model_without_ddp.cnn(images)
                        x_swin = F.interpolate(images, size=(model_without_ddp.img_size, model_without_ddp.img_size)) \
                            if images.size(2) != model_without_ddp.img_size or images.size(3) != model_without_ddp.img_size else images
                        swin_features_list = model_without_ddp.swin(x_swin)
                        swin_features = swin_features_list[0] if isinstance(swin_features_list, list) else swin_features_list
                        swin_features = F.interpolate(swin_features, size=cnn_features.shape[2:])
                        fused_features = torch.cat([cnn_features, swin_features], dim=1)
                        fused_features = model_without_ddp.fusion(fused_features)
                    return fused_features
                model_without_ddp.prototype_layer.project_prototypes(data_loader_train, feature_extractor)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, scaler)

        # ---- Validation after each epoch ----
        acc1, acc5, loss = validate(config, data_loader_val, model, is_validation=True, args=args)
        logger.info(f"Mean Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.2f}%")
        logger.info(f"Mean Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Validation Max mean accuracy: {max_accuracy:.2f}%')

        # ---- XAI visualization (every xai-vis-freq epochs) ----
        if args.xai and (epoch % args.xai_vis_freq == 0):
            logger.info("Generating XAI visualizations...")
            grad_cam = GradCAM(model_without_ddp)
            for i, (samples, targets) in enumerate(data_loader_val):
                if i >= 5: break
                samples = samples.cuda(non_blocking=True)
                outputs = model_without_ddp(samples)
                logits = outputs['logits']
                target_class = torch.argmax(logits, dim=1)
                cam_dict = grad_cam.generate_cam(samples, target_class=target_class)
                grad_cam.visualize(cam_dict, save_path=os.path.join(config.OUTPUT, f'cam_epoch{epoch}_sample{i}.png'))
            if hasattr(model_without_ddp, 'prototype_layer'):
                visualize_prototypes(model_without_ddp, data_loader_train.dataset, save_dir=os.path.join(config.OUTPUT, f'prototypes_epoch{epoch}'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # ---- Test ONCE after all training ----
    logger.info("Training complete. Running final test evaluation...")
    acc1, acc5, loss = validate(config, data_loader_test, model, is_validation=False, args=args)
    logger.info(f"Test set: Accuracy: {acc1:.2f}%  Loss: {loss:.5f}")

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()  # Ensure targets shape [B, 14] and float

        # Remove mixup for multi-label unless you have a multi-label compatible mixup
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with autocast():
            outputs = model(samples)
            if config.MODEL.TYPE == 'hybrid_swin_cnn':
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs[0], targets)
                for i in range(1, len(targets)):
                    loss += criterion(outputs[i], targets[i])

        scaler.scale(loss).backward()
        if config.TRAIN.CLIP_GRAD:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            print(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    lr = optimizer.param_groups[0]['lr']
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    print(
        f'Train: [{epoch}/{config.TRAIN.EPOCHS}]\t'
        f'lr {lr:.6f}\t'
        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(config, data_loader, model, is_validation, args=None):
    valid_or_test = "Validation" if is_validation else "Test"
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    all_targets = []
    all_outputs = []

    end = time.time()
    for idx, (images, targets) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()  # [B, 14]
        output = model(images)
        logits = output['logits']

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss_meter.update(loss.item(), images.size(0))

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == targets).float().mean()
        acc1_meter.update(correct.item(), images.size(0))

        all_targets.append(targets.cpu().numpy())
        all_outputs.append(torch.sigmoid(logits).cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'{valid_or_test}: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB'
            )

    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    aucs = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
        except Exception:
            auc = float('nan')
        aucs.append(auc)
    from statistics import mean
    print(f'{valid_or_test} MEAN AUC: {mean([a for a in aucs if not np.isnan(a)]):.5f}')

    if args is not None and args.xai and valid_or_test == "Test":
        print("Generating XAI visualizations on test set...")
        grad_cam = GradCAM(model.module if hasattr(model, 'module') else model)
        for i, (samples, targets) in enumerate(data_loader):
            if i >= 5: break
            samples = samples.cuda(non_blocking=True)
            outputs = model.module(samples) if hasattr(model, 'module') else model(samples)
            logits = outputs['logits']
            target_class = torch.argmax(logits, dim=1)
            cam_dict = grad_cam.generate_cam(samples, target_class=target_class)
            grad_cam.visualize(cam_dict, save_path=os.path.join(config.OUTPUT, f'cam_test_sample{i}.png'))
        if hasattr(model, 'prototype_layer'):
            visualize_prototypes(model, data_loader.dataset, save_dir=os.path.join(config.OUTPUT, f'prototypes_test'))

    return acc1_meter.avg, 0, loss_meter.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())

    main(config, args)
