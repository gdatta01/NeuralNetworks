from yacs.config import CfgNode


def network_config():
    cfg = CfgNode()
    cfg.INPUTS = 0
    cfg.HIDDEN_LAYERS = []
    cfg.OUTPUTS = 0
    return cfg


def datset_config():
    cfg = CfgNode()
    cfg.NAME = ""
    cfg.PATH = ""
    cfg.NORMALIZE = False
    return cfg


def training_config():
    cfg = CfgNode()
    cfg.TARGET_SMOOTHING = 0.0
    cfg.LOSS = ""
    cfg.EPOCHS = 0
    cfg.BATCH_SIZE = 0
    cfg.LR = 0.0
    cfg.LR_SCHEDULE = ""
    cfg.MOMENTUM = 0.0
    cfg.WEIGHT_DECAY = 0.0
    cfg.HOLDOUT = 0.0
    return cfg


_C = CfgNode()

_C.NETWORK = network_config()

_C.DATASET = datset_config()

_C.TRAINING = training_config()

cfg = _C