import os
import argparse
from tqdm import tqdm
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from lib.config import cfg, update_config
from lib.utils import get_logger, get_model, get_optimizer, get_scheduler, get_dataset, get_criterion, reduce_tensor
import torch

def train_step(inputs, labels, model, criterion, device):
    model.train()  # Set model to training mode

    inputs = inputs.to(device)
    labels = labels.to(device).long()

    # zero the parameter gradients
    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        # statistics
    return reduce_tensor(loss * inputs.size(0)).item(), reduce_tensor(torch.sum(preds == labels.data)).item() / args.world_size


def val_step(val_dataloader, model, device):
    model.eval()
    val_running_corrects = 0.
    with torch.set_grad_enabled(False):
        for data in tqdm(val_dataloader, desc='val'):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data).item()

    val_running_corrects = reduce_tensor(
        torch.tensor([val_running_corrects], dtype=torch.int64, device=args.local_rank % args.world_size)).item() / args.world_size
    val_epoch_acc = float(val_running_corrects) / len(val_dataloader.dataset)
    return val_epoch_acc


def save_checkpoints(model, path):
    if args.rank == 0:
        torch.save(
            model.state_dict(),
            path
        )


def train_model(
        train_dataloder,
        val_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        session,
        batch_size
):
    device = torch.device('cuda')

    os.makedirs(session.SAVEPATH, exist_ok=True)
    checkpoints_path = os.path.join(session.SAVEPATH, session.NAME)
    os.makedirs(checkpoints_path, exist_ok=True)

    for epoch in range(session.MAX_EPOCH):
        if args.local_rank == 0:
            logger.info('Epoch {}/{}'.format(epoch, session.MAX_EPOCH - 1))
            logger.info('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()

        running_loss = 0.0
        running_corrects = 0
        train_dataloader.sampler.set_epoch(epoch)
        # Iterate over data.
        with tqdm(total=len(train_dataloder.dataset), desc='Iterate over data') as pbar:
            for step, (inputs, labels) in enumerate(train_dataloder):
                if step % session.VAL_STEP == 0:
                    val_acc = val_step(val_dataloader, model, device)
                    if args.local_rank == 0:
                        logger.info('val Acc: {:.4f}'.format(val_acc))

                    save_checkpoints(model, os.path.join(checkpoints_path, 'epoch_{}_step_{}_acc:{:.4f}.pth'.format(epoch, step * batch_size, val_acc)))

                step_running_loss, step_running_corrects = train_step(inputs, labels, model, criterion, device)

                running_loss += step_running_loss
                running_corrects += step_running_corrects
                if step % session.SHOW_STEP == 0 and args.local_rank == 0:
                    count = (step + 1) * batch_size
                    logger.info("train Loss:{:.4f} Acc:{:.4f}".format(
                        float(running_loss) / count,
                        float(running_corrects) / count)
                    )
                if step % session.SAVE_STEP == 0:
                    save_checkpoints(model, os.path.join(checkpoints_path, 'latest.pth'))
                pbar.update(batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, help='config yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    logger, log_file = get_logger(cfg)

    torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
    torch.distributed.init_process_group(
        backend='nccl')  # , init_method='tcp://localhost:23456', rank=0, world_size=1)
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    logger.info('current process local rank %d' % args.local_rank)
    logger.info('current process rank %d' % args.rank)
    logger.info('total world size %d' % args.world_size)

    train_dataloader = get_dataset(cfg.TRAIN_DATA)
    val_dataloader = get_dataset(cfg.VAL_DATA)

    model = get_model(cfg.MODEL)

    model.to(torch.device('cuda'))
    model = DDP(model, delay_allreduce=True)

    if cfg.SESSION.RESUME:
        resume_file_path = os.path.join(cfg.SESSION.SAVEPATH, cfg.SESSION.RESUME)
        if os.path.isfile(resume_file_path):
            logger.info("loading checkpoint '{}'".format(resume_file_path))
            model.load_state_dict(torch.load(resume_file_path, map_location=lambda storage, loc: storage.cuda(
                args.local_rank % args.world_size)))
            if cfg.MODEL.CLASSIFIER.REINIT:
                torch.nn.init.xavier_normal_(model.last_linear.weights)
                model.last_linear.bias.data.zero_()
        else:
            logger.info("=> no checkpoint found at '{}'".format(resume_file_path))


    if args.local_rank == 0:
        logger.info('train data size:', len(train_dataloader.dataset))
        logger.info('val data size:', len(val_dataloader.dataset))
        logger.info('model:', model)

    criterion = get_criterion(cfg.CRITERION)
    optimizer = get_optimizer(cfg.OPTIMIZER, model)
    lr_scheduler = get_scheduler(cfg.LR_SCHEDULER, optimizer)

    model = train_model(
        train_dataloder=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        session=cfg.SESSION,
        batch_size=cfg.TRAIN_DATA.BATCHSIZE,
    )
