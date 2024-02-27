from typing import Optional, Dict, List, Union
import os
import glob
import json

import numpy as np

from quickstats import semistaticmethod

from keywords import FeatureLevel, HIGH_LEVEL, LOW_LEVEL
from keywords import TRAIN_FEATURES
from keywords import MassOrdering, MASS_UNORDERED, MASS_ORDERED
from keywords import ModelType, DEDICATED_SUPERVISED, PARAM_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY
from keywords import MLP_LAYERS

from base_loader import BaseLoader



class ModelLoader(BaseLoader):

    def __init__(self, feature_level:str,
                 mass_ordering:bool=MASS_UNORDERED,
                 variables:Optional[str]=None,
                 distributed:bool=True,
                 strategy=None,
                 verbosity:str='INFO'):
        super().__init__(feature_level=feature_level,
                         mass_ordering=mass_ordering,
                         variables=variables,
                         distributed=distributed,
                         strategy=strategy,
                         verbosity=verbosity)

    def get_supervised_model_inputs(self, feature_metadata:Dict,
                                    downcast:bool=True):
        from tensorflow.keras.layers import Input
        label_map = {
            'part_coords': 'points',
            'part_features': 'features',
            'part_masks': 'masks',
            
        }
        tmp_metadata = feature_metadata.copy()
        if downcast:
            for metadata in tmp_metadata.values():
                if metadata['dtype'] == 'float64':
                    metadata['dtype'] = 'float32'
        if self.variables is not None:
            nvar = len(self.variables.split(","))
            tmp_metadata['jet_features']['shape'][-1] = nvar
        inputs = {}
        for feature, metadata in tmp_metadata.items():
            key = label_map.get(feature, feature)
            inputs[key] = Input(**metadata, name=feature)
        return inputs

    def get_train_config(self, checkpoint_dir:str,
                         model_type:Optional[ModelType]=None):
        if self.feature_level == HIGH_LEVEL:
            epochs = 100
            patience = 10
        elif self.feature_level == LOW_LEVEL:
            epochs = 20
            patience = 5
        else:
            raise RuntimeError(f'unknown feature level: {self.features_level.key}')
        config = {
            # for binary classification
            'loss'       : 'binary_crossentropy',
            'metrics'    : ['accuracy'],
            'epochs'     : epochs,
            'optimizer'  : 'Adam',
            'optimizer_config': {
                'learning_rate': 0.001
            },
            'checkpoint_dir': checkpoint_dir,
            'callbacks': {
                'lr_scheduler': {
                    'initial_lr': 0.001,
                    'lr_decay_factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-6
                },
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': patience,
                    'restore_best_weights': True
                },
                'model_checkpoint':{
                    'save_weights_only': True,
                    # save model checkpoint every epoch
                    'save_freq': 'epoch'
                },
                'metrics_logger':{
                    'save_freq': -1
                },
                'weights_logger':{
                    'save_freq': -1,
                    'display_weight': True
                }
            }
        }

        if (model_type is not None) and (ModelType.parse(model_type) == SEMI_WEAKLY):
            if self.feature_level == HIGH_LEVEL:
                config['callbacks']['lr_scheduler'] = {
                    'initial_lr': 0.1,
                    'lr_decay_factor': 0.5,
                    'patience': 10,
                    'min_lr': 1e-6                    
                }
    
        return config    
    
    def _get_high_level_model(self, feature_metadata:Dict,
                              parametric:bool=True):
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization
        import tensorflow as tf

        all_inputs = self.get_supervised_model_inputs(feature_metadata)

        x1 = all_inputs['jet_features']
        if parametric:
            param_feature = self._get_param_feature()
            x2 = all_inputs[param_feature]
            inputs = [x1, x2]
            # concatenate the input features and physics parameters
            x = tf.concat([x1, tf.expand_dims(x2, axis=-1)], -1)
            # flatten the inputs
            x = tf.reshape(x, (-1, tf.reduce_prod(tf.shape(x)[1:])))
        else:
            inputs = [x1]
            x = tf.reshape(x1, (-1, tf.reduce_prod(tf.shape(x1)[1:])))
            
        layers = list(MLP_LAYERS)
        for nodes, activation in layers:
            x = Dense(nodes, activation)(x)
            
        model = Model(inputs=inputs, outputs=x, name='HighLevel')
        
        return model
    
    def _get_low_level_model(self, feature_metadata:Dict,
                             parametric:bool=True):
        from aliad.interface.tensorflow.models import MultiParticleNet
        all_inputs = self.get_supervised_model_inputs(feature_metadata)
        keys = ['points', 'features', 'masks', 'jet_features']
        if parametric:
            param_feature = self._get_param_feature()
            all_inputs['param_features'] = all_inputs[param_feature]
            keys.append('param_features')
        inputs = {key: all_inputs[key] for key in keys}
        model_builder = MultiParticleNet()
        model = model_builder.get_model(**inputs)
        return model

    def _distributed_wrapper(self, fn, **kwargs):
        import tensorflow as tf
        if self.distributed:
            strategy = self.distribute_strategy
            with strategy.scope():
                result = fn(**kwargs)
        else:
            result = fn(**kwargs)
        return result
            
    def get_supervised_model(self, feature_metadata:Dict, parametric:bool):
        if self.feature_level == HIGH_LEVEL:
            model_fn = self._get_high_level_model
        elif self.feature_level == LOW_LEVEL:
            model_fn = self._get_low_level_model

        kwargs = {
            'feature_metadata': feature_metadata,
            'parametric': parametric
        }
        return self._distributed_wrapper(model_fn, **kwargs)

    @staticmethod
    def get_single_parameter_model(activation:str='relu',
                                   exponential:bool=False,
                                   kernel_initializer=None):
        import tensorflow as tf
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Dense
        inputs = Input(shape=(1,))
        outputs = Dense(1, use_bias=False,
                        activation=activation,
                        kernel_initializer=kernel_initializer)(inputs)
        if exponential:
            outputs = tf.exp(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model    

    @semistaticmethod
    def get_semi_weakly_weights(self, m1:float, m2:float,
                                mu:Optional[float]=None,
                                alpha:Optional[float]=None):
        import tensorflow as tf
        weights = {
            'm1': self.get_single_parameter_model(kernel_initializer=tf.constant_initializer(float(m1))),
            'm2': self.get_single_parameter_model(kernel_initializer=tf.constant_initializer(float(m2))),
        }
        if mu is not None:
            weights['mu'] = self.get_single_parameter_model(activation='linear',
                                                            exponential=True,
                                                            kernel_initializer=tf.constant_initializer(float(mu)))
        if alpha is not None:
            weights['alpha'] = self.get_single_parameter_model(kernel_initializer=tf.constant_initializer(float(alpha)))
            
        return weights

    @staticmethod
    def _get_one_signal_semi_weakly_layer(fs_out, mu,
                                          epsilon:float=1e-5):
        LLR = fs_out / (1. - fs_out + epsilon)
        LLR_xs = 1. + mu * (LLR - 1.)
        ws_out = LLR_xs / (1 + LLR_xs)
        return ws_out

    @staticmethod
    def _get_two_signal_semi_weakly_layer(fs_2_out, fs_3_out, mu,
                                          alpha, epsilon:float=1e-5):
        LLR_2 = fs_2_out / (1. - fs_2_out + epsilon)
        LLR_3 = fs_3_out / (1. - fs_3_out + epsilon)
        LLR_xs = 1. + mu * (alpha * LLR_3 + (1 - alpha) * LLR_2 - 1.)
        ws_out = LLR_xs / (1 + LLR_xs)
        return ws_out    

    def _get_semi_weakly_model(self, feature_metadata:Dict,
                               fs_model_path:str,
                               m1:float=300., m2:float=300.,
                               mu:float=-3.,
                               alpha:float=0.5,
                               fs_model_path_2:Optional[str]=None,
                               mass_scale:float=100.,
                               epsilon:float=1e-5):
        import tensorflow as tf
        inputs = self.get_supervised_model_inputs(feature_metadata)
        weights = self.get_semi_weakly_weights(m1=m1, m2=m2, mu=mu, alpha=alpha)
        m1_out = float(mass_scale) * weights['m1'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        m2_out = float(mass_scale) * weights['m2'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        mu_out = weights['mu'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        alpha_out = weights['alpha'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        mass_params = tf.keras.layers.concatenate([m1_out, m2_out])

        train_features = list(TRAIN_FEATURES[self.feature_level])
        train_inputs = [inputs[feature] for feature in train_features]
        fs_inputs = [inputs[feature] for feature in train_features]
        fs_inputs.append(mass_params)

        fs_model = self.load_model(fs_model_path)
        self.freeze_all_layers(fs_model)
        # single signal
        if fs_model_path_2 is None:
            fs_out = fs_model(fs_inputs)
            ws_out = self._get_one_signal_semi_weakly_layer(fs_out, mu=mu_out,
                                                            epsilon=epsilon)
        else:
            fs_2_model = fs_model
            fs_3_model = self.load_model(fs_model_path_2)
            self.freeze_all_layers(fs_3_model)
            fs_2_out = fs_2_model(fs_inputs)
            fs_3_out = fs_3_model(fs_inputs)
            ws_out = self._get_one_signal_semi_weakly_layer(fs_2_out, fs_3_out,
                                                            mu=mu_out,
                                                            alpha=alpha_out,
                                                            epsilon=epsilon)
        ws_model = tf.keras.Model(inputs=train_inputs, outputs=ws_out)  
        return ws_model

    def get_semi_weakly_model(self, feature_metadata:Dict,
                              fs_model_path:str,
                              m1:float=300., m2:float=300.,
                              mu:float=-3,
                              alpha:float=0.5,
                              fs_model_path_2:Optional[str]=None,
                              mass_scale:float=100.,
                              epsilon:float=1e-5):
        kwargs = {
            'feature_metadata': feature_metadata,
            'fs_model_path': fs_model_path,
            'm1': m1,
            'm2': m2,
            'mu': mu,
            'alpha': alpha,
            'fs_model_path_2': fs_model_path_2,
            'mass_scale': mass_scale,
            'epsilon': epsilon
        }
        model_fn = self._get_semi_weakly_model
        return self._distributed_wrapper(model_fn, **kwargs)

    @staticmethod
    def set_semi_weakly_model_weights(ws_model, m1:float, m2:float,
                                      mu:float, alpha:Optional[float]=None):
        import tensorflow as tf
        ws_model.trainable_weights[0].assign(tf.fill((1, 1), float(m1)))
        ws_model.trainable_weights[1].assign(tf.fill((1, 1), float(m2)))
        ws_model.trainable_weights[2].assign(tf.fill((1, 1), float(mu)))
        if (alpha is not None):
            ws_model.trainable_weights[3].assign(tf.fill((1, 1), float(alpha)))
        
    @staticmethod
    def compile_model(model, config:Dict):
        import tensorflow as tf
        optimizer = tf.keras.optimizers.get({'class_name': config['optimizer'],
                                             'config': config['optimizer_config']})
        model.compile(loss=config['loss'],
                      optimizer=optimizer,
                      metrics=config['metrics'])

    @staticmethod
    def load_model(model_path:str):
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)

    @staticmethod
    def freeze_all_layers(model):
        # equivalent to freeze_model
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def freeze_model(model):
        model.trainable = False

    @staticmethod
    def get_callbacks(model_type:ModelType, config:Dict):
        model_type = ModelType.parse(model_type)
        from aliad.interface.tensorflow.callbacks import (LearningRateScheduler, MetricsLogger,
                                                          WeightsLogger, EarlyStopping)
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        checkpoint_dir = config['checkpoint_dir']
        
        lr_scheduler = LearningRateScheduler(**config['callbacks']['lr_scheduler'])
        early_stopping = EarlyStopping(**config['callbacks']['early_stopping'])
        model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'model_weights_epoch_{epoch:02d}.h5'),
                                           **config['callbacks']['model_checkpoint'])
        metrics_logger = MetricsLogger(checkpoint_dir, **config['callbacks']['metrics_logger'])
    
        callbacks = {
            'lr_scheduler': lr_scheduler,
            'early_stopping': early_stopping,
            'model_checkpoint': model_checkpoint,
            'metrics_logger': metrics_logger
        }
    
        if model_type == SEMI_WEAKLY:
            weights_logger = WeightsLogger(checkpoint_dir, **config['callbacks']['weights_logger'])
            callbacks['weights_logger'] = weights_logger

        return callbacks

    @staticmethod
    def restore_model(early_stopping, model, checkpoint_dir:str):
        metrics_ckpt_filepath = os.path.join(checkpoint_dir, "epoch_metrics",
                                             "metrics_epoch_{epoch}.json")
        model_ckpt_filepath = os.path.join(checkpoint_dir,
                                           "model_weights_epoch_{epoch:02d}.h5")
        early_stopping.restore(model, metrics_ckpt_filepath=metrics_ckpt_filepath,
                               model_ckpt_filepath=model_ckpt_filepath)

    @staticmethod
    def get_model_save_path(checkpoint_dir:str):
        return os.path.join(checkpoint_dir, "full_train.keras")