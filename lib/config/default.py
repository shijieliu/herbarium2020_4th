from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from yacs.config import CfgNode as CN


_C = CN()

# ----- basic SETTINGS -----
_C.SESSION = CN()
_C.SESSION.NAME = 'default'
_C.SESSION.SAVE_STEP = 1
_C.SESSION.SHOW_STEP = 1
_C.SESSION.VAL_STEP = 1
_C.SESSION.MAX_EPOCH = 1
_C.SESSION.SAVEPATH = ''
_C.SESSION.RESUME = 'latest'


# ----- Model SETTINGS -----
_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.TYP = 'se_resnext50_32x4d'
_C.MODEL.BACKBONE.FREEZED = False
_C.MODEL.CLASSIFIER = CN()
_C.MODEL.CLASSIFIER.CATEGORY_NUM = 0
_C.MODEL.CLASSIFIER.BIAS = True
_C.MODEL.CLASSIFIER.REINIT = False

# ----- criterion SETTINGS -----
_C.CRITERION = CN()
_C.CRITERION.TYP = ''

# ----- optimizer SETTINGS -----
_C.OPTIMIZER = CN()
_C.OPTIMIZER.BASE_LR = 0.1
_C.OPTIMIZER.TYP = 'SGD'
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4
_C.OPTIMIZER.MOMENTUM = 0.4
_C.OPTIMIZER.NESTEROV = False

# ----- LR_SCHEDULER SETTINGS -----
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYP = 'default'
_C.LR_SCHEDULER.LR_STEP = []
_C.LR_SCHEDULER.LR_FACTOR = 0.1

# ----- TRAIN DATA SETTINGS -----
_C.TRAIN_DATA = CN()
_C.TRAIN_DATA.PATH = ''
_C.TRAIN_DATA.BATCHSIZE = 1
_C.TRAIN_DATA.NUM_WORKERS = 4
_C.TRAIN_DATA.SAMPLER = 'default'
_C.TRAIN_DATA.SHUFFLE = True
_C.TRAIN_DATA.TRANSFORMS = []


# ----- VAL DATA SETTINGS -----
_C.VAL_DATA = CN()
_C.VAL_DATA.PATH = ''
_C.VAL_DATA.BATCHSIZE = 1
_C.VAL_DATA.NUM_WORKERS = 4
_C.VAL_DATA.SAMPLER = 'default'
_C.VAL_DATA.SHUFFLE = False
_C.VAL_DATA.TRANSFORMS = []

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    cfg.freeze()