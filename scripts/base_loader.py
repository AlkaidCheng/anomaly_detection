from typing import Optional, Dict, List, Union
import os
import glob
import json

import numpy as np

from keywords import FeatureLevel, HIGH_LEVEL, LOW_LEVEL
from keywords import MassOrdering, MASS_UNORDERED, MASS_ORDERED
from keywords import TRAIN_FEATURES, WEIGHT_FEATURES, PARAM_FEATURES
from keywords import ModelType, DEDICATED_SUPERVISED, PARAM_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY

from quickstats import AbstractObject

class BaseLoader(AbstractObject):

    @property
    def feature_level(self):
        return self._feature_level

    @feature_level.setter
    def feature_level(self, value:str):
        feature_level = FeatureLevel.parse(value)
        self._feature_level = feature_level

    @property
    def mass_ordering(self):
        return self._mass_ordering

    @mass_ordering.setter
    def mass_ordering(self, value:str):
        mass_ordering = MassOrdering.parse(value)
        self._mass_ordering = mass_ordering

    def __init__(self, feature_level:str,
                 mass_ordering:bool=MASS_UNORDERED,
                 variables:Optional[str]=None,
                 verbosity:str='INFO'):
        super().__init__(verbosity=verbosity)
        self.feature_level = feature_level
        self.mass_ordering = mass_ordering
        self.variables = variables

    def _get_weight_feature(self):
        return WEIGHT_FEATURES[self.mass_ordering]

    def _get_param_feature(self):
        return PARAM_FEATURES[self.mass_ordering]


    def _get_train_features(self, model_type:ModelType):
        model_type = ModelType.parse(model_type)
        features = list(TRAIN_FEATURES[self.feature_level])
        if model_type == PARAM_SUPERVISED:
            param_feature = self._get_param_feature()
            features.append(param_feature)
        return features

    def _get_aux_features(self, model_type:ModelType):
        model_type = ModelType.parse(model_type)
        features = ['label']
        if model_type in [IDEAL_WEAKLY, SEMI_WEAKLY]:
            return features
        if self.weighted:
            weight_feature = self._get_weight_feature()
            features.append(weight_feature)
        return features
        
    def _get_required_features(self, model_type:ModelType):
        model_type = ModelType.parse(model_type)
        train_features = self._get_train_features(model_type=model_type)
        aux_features = self._get_aux_features(model_type=model_type)
        features = train_features + aux_features
        return features
