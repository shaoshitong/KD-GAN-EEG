from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']
_C.EPOCH = 100
_C.WARMUP_EPOCH=20
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten by command line argument
# _C.DATA.DEAP_DATA_PATH = '/data/EEG/Channel_DE_sample/Binary/'
# _C.DATA.SEED_DATA_PATH = '/data/EEG/DE_4D/'
_C.DATA.DEAP_DATA_PATH = 'G:/Alex/DEAP_experiment/sample_data/DE_4D_channel/Binary/'
_C.DATA.SEED_DATA_PATH = 'G:/Alex/SEED_experiment/Three sessions sample/DE_4D/'
_C.DATA.DATASET = 'seed'
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
_C.MODEL.IF_TURN_LR=True

_C.MODEL.DEAP = CN()
_C.MODEL.DEAP.BATCH_SIZE=96
_C.MODEL.DEAP.IN_FEATURES=120*128
_C.MODEL.DEAP.HIDDEN_DIM=512
_C.MODEL.DEAP.OUT_FEATURES=2048
_C.MODEL.DEAP.CRITIC=5
_C.MODEL.DEAP.NUM_CLASSES=2
_C.MODEL.SEED = CN()
_C.MODEL.SEED.BATCH_SIZE=96
_C.MODEL.SEED.IN_FEATURES=310*180
_C.MODEL.SEED.CRITIC=5
_C.MODEL.SEED.HIDDEN_DIM=512
_C.MODEL.SEED.OUT_FEATURES=2048
_C.MODEL.SEED.NUM_CLASSES=3

def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    return config


