import logging
import os
import time
import torch
import torch.distributed as dist
from .data.data import HerbariumDS, DataLoaderX, make_weights_for_balanced_classes, DistributedWeightedSampler
import torch.nn as nn
from .network import senet as backbonezoo


def get_logger(cfg):

    log_dir = os.path.join(cfg.SESSION.SAVEPATH, cfg.SESSION.NAME)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}.log".format(time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file

def get_sampler(cfg, dataset):
    if cfg.SAMPLER == 'default':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=cfg.SHUFFLE)  # use set_epoch to shuffle the data
    elif cfg.SAMPLER == 'reverse':
        weights = make_weights_for_balanced_classes(dataset.data)
        weights = torch.DoubleTensor(weights)
        sampler = DistributedWeightedSampler(dataset, weights)
    else:
        raise NotImplementedError('sampler %s not impleted' % cfg.SAMPLER)
    return sampler

def get_dataset(cfg):
    dataset = HerbariumDS(cfg.PATH, cfg.TRANSFORMS)
    return DataLoaderX(dataset, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, sampler=get_sampler(cfg, dataset))


def get_criterion(cfg):
    if cfg.TYP == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('criterion typ %s not implemented' % cfg.TYP)


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def get_model(cfg):
    backbone = getattr(backbonezoo, cfg.BACKBONE.TYP)()
    if cfg.BACKBONE.FREEZED:
        for params in backbone.parameters():
            params.requires_grad = False
    else:
        for params in backbone.parameters():
            params.requires_grad = True

    fc_inputs = backbone.last_linear.in_features
    backbone.last_linear = nn.Linear(fc_inputs, cfg.CLASSIFIER.CATEGORY_NUM, bias=cfg.CLASSIFIER.BIAS)
    return backbone


def get_scheduler(cfg, optimizer):
    if cfg.TYP == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.LR_STEP,
            gamma=cfg.LR_FACTOR,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_optimizer(cfg, model):
    base_lr = cfg.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.TYP == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            nesterov=cfg.NESTEROV,
        )
    elif cfg.TYP == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError('optimizer %s not impleted' % cfg.TYP)
    return optimizer
