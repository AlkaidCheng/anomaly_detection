import os
from quickstats.core import GeneralEnum, DescriptiveEnum

class KeyDescriptiveEnum(DescriptiveEnum):
    def __new__(cls, value: int, description: str = "", key: str = ""):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.key = key
        return obj

class FeatureLevel(KeyDescriptiveEnum):
    HIGH_LEVEL = (0, "High level jet features", "high_level")
    LOW_LEVEL  = (1, "Low level particle features + high level jet features", "low_level")
HIGH_LEVEL = FeatureLevel.HIGH_LEVEL
LOW_LEVEL  = FeatureLevel.LOW_LEVEL

class DecayMode(KeyDescriptiveEnum):
    __aliases__ = {
        "qq" : "two_prong",
        "qqq": "three_prong"
    }
    TWO_PRONG   = (0, "Two-prong decay (X(qq)Y(qq))", "qq")
    THREE_PRONG = (1, "Three-prong decay (X(qqq)Y(qq))", "qqq")
TWO_PRONG   = DecayMode.TWO_PRONG
THREE_PRONG = DecayMode.THREE_PRONG

class ModelType(KeyDescriptiveEnum):
    DEDICATED_SUPERVISED = (0, "Supervised training at a dedicated mass point", "dedicated_supervised")
    PARAM_SUPERVISED     = (1, "Supervised training with parametric masses", "param_supervised")
    IDEAL_WEAKLY         = (2, "Ideal weakly supervised training", "ideal_weakly")
    SEMI_WEAKLY          = (3, "Semi-weakly supervised training", "semi_weakly")
DEDICATED_SUPERVISED = ModelType.DEDICATED_SUPERVISED
PARAM_SUPERVISED = ModelType.PARAM_SUPERVISED
IDEAL_WEAKLY = ModelType.IDEAL_WEAKLY
SEMI_WEAKLY = ModelType.SEMI_WEAKLY

class MassOrdering(KeyDescriptiveEnum):
    MASS_UNORDERED = (0, "Unordered mass parameters", "mass_unordered")
    MASS_ORDERED   = (1, "Ordered mass parameters (m1 >= m2)", "mass_ordered")
MASS_UNORDERED = MassOrdering.MASS_UNORDERED
MASS_ORDERED = MassOrdering.MASS_ORDERED

NUM_SHARDS = 100
BASE_SEED = 2023

# sample formats
SIG_SAMPLE = 'W_{decay_mode}'
BKG_SAMPLE = 'QCD_qq'
EXT_BKG_SAMPLE = 'extra_QCD_qq'
PARAM_SIG_SAMPLE = 'W_{decay_mode}_{m1}_{m2}'
PARAM_BKG_SAMPLE = 'QCD_qq_{m1}_{m2}'
PARAM_EXT_BKG_SAMPLE = 'extra_QCD_qq_{m1}_{m2}'

# ordering is important
TRAIN_FEATURES = {
    HIGH_LEVEL: ['jet_features'],
    LOW_LEVEL: ['part_coords', 'part_features',
                'part_masks', 'jet_features']
}

WEIGHT_FEATURES = {
    MASS_UNORDERED: 'weight',
    MASS_ORDERED: 'weight_merged'
}

PARAM_FEATURES = {
    MASS_UNORDERED: 'param_masses_unordered',
    MASS_ORDERED: 'param_masses_ordered'
}

MLP_LAYERS = [(256, 'relu'),
              (128, 'relu'),
              (64, 'relu'),
              (1, 'sigmoid')]

# paths
DATASET_INDEX_PATH = ("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                      "LHCO_RnD/tfrecords/dataset_indices.json")
DATASET_DIR = ("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
               "LHCO_RnD/tfrecords")
BASE_OUTDIR = "/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD_new"
MODEL_OUTDIRS = {
    DEDICATED_SUPERVISED : os.path.join(BASE_OUTDIR, "fully_supervised"),
    PARAM_SUPERVISED     : os.path.join(BASE_OUTDIR, "fully_supervised"),
    IDEAL_WEAKLY         : os.path.join(BASE_OUTDIR, "ideal_weakly"),
    SEMI_WEAKLY          : os.path.join(BASE_OUTDIR, "semi_weakly")
}