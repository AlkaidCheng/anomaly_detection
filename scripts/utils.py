from typing import Optional, Dict, List

import os
import glob
import json

import numpy as np

def initial_check():
    import tensorflow as tf
    import aliad as ad
    print(f'aliad version : {ad.__version__}')
    print(f'tensorflow version  : {tf.__version__}')
    os.system("nvidia-smi")
    os.system("nvcc --version")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass

def get_model_inputs(feature_metadata, variables=None):
    from tensorflow.keras.layers import Input
    tmp_metadata = feature_metadata.copy()
    if variables is not None:
        nvar = len(variables.split(","))
        tmp_metadata['jet_features']['shape'] = [2, nvar]
    inputs = {
        'part_coords': Input(shape=tmp_metadata['part_coords']['shape']),
        'part_features': Input(shape=tmp_metadata['part_features']['shape']),
        'part_masks': Input(**tmp_metadata['part_masks']),
        'jet_features': Input(tmp_metadata['jet_features']['shape'])
    }
    return inputs



def get_high_level_model(feature_metadata, parametric:bool=True, mass_ordering:bool=True,
                         variables=None, batchnom:bool=True):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization
    import tensorflow as tf
    
    if variables is not None:
        nvar = len(variables.split(","))
        feature_metadata['jet_features']['shape'] = [2, nvar]
    x1 = Input(**feature_metadata['jet_features'])
    
    if parametric:
        label = 'param_masses_ordered' if mass_ordering else 'param_masses_unordered'
        x2 = Input(**feature_metadata[label])
        inputs = [x1, x2]
        # concatenate the input features and physics parameters
        x = tf.concat([x1, tf.expand_dims(x2, axis=-1)], -1)
        # flatten the inputs
        x = tf.reshape(x, (-1, tf.reduce_prod(tf.shape(x)[1:])))
    else:
        inputs = x1
        x = tf.reshape(x1, (-1, tf.reduce_prod(tf.shape(x1)[1:])))
        
    layers = [(256, 'relu'),
              (128, 'relu'),
              (64, 'relu'),
              (1, 'sigmoid')]
    
    for nodes, activation in layers:
        x = Dense(nodes, activation)(x)
        #if batchnom:
        #    x = BatchNormalization()(x)
        
    model = Model(inputs=inputs, outputs=x, name='HighLevel')
    
    return model

def get_low_level_model(feature_metadata, parametric:bool=True, mass_ordering:bool=True,
                        variables=None):
    from tensorflow.keras.layers import Input
    from aliad.interface.tensorflow.models import MultiParticleNet
    inputs = {
        'points': Input(shape=feature_metadata['part_coords']['shape']),
        'features': Input(shape=feature_metadata['part_features']['shape']),
        'masks': Input(**feature_metadata['part_masks']),
        'jet_features': Input(shape=feature_metadata['jet_features']['shape'])
    }
    if parametric:
        label = 'param_masses_ordered' if mass_ordering else 'param_masses_unordered'
        inputs['param_features'] = Input(shape=feature_metadata[label]['shape'])
    model_builder = MultiParticleNet()
    model = model_builder.get_model(**inputs)
    return model

def get_required_features(supervised:bool, high_level:bool,
                          parametric:bool, mass_ordering:bool,
                          weighted:bool):
    features = []
    if high_level:
        features.append('jet_features')
    else:
        features.extend(['part_coords', 'part_features',
                         'part_masks', 'jet_features'])
    if supervised:
        if parametric:
           if mass_ordering:
               features.append('param_masses_ordered')
           else:
               features.append('param_masses_unordered')
        features.append('label')
        if weighted:
            if mass_ordering:
                features.append('weight_merged')
            else:
                features.append('weight')
    return features


# method to extract the feature vectors and label from the tfrecord
def get_input_fn(high_level:bool=True, parametric:bool=True, mass_ordering:bool=False,
                 weighted:bool=True, variables=None):
    import tensorflow as tf
    if high_level:
        if variables is None:
            if parametric:
                if mass_ordering:
                    if weighted:
                        return lambda X: ((X['jet_features'], X['param_masses_ordered']), X['label'], X['weight_merged'][0])
                    else:
                        return lambda X: ((X['jet_features'], X['param_masses_ordered']), X['label'])
                else:
                    if weighted:
                        return lambda X: ((X['jet_features'], X['param_masses_unordered']),
                                          X['label'], X['weight'][0])
                    else:
                        return lambda X: ((X['jet_features'], X['param_masses_unordered']), X['label'])
            else:
                if weighted:
                    return lambda X: (X['jet_features'], X['label'], X['weight'][0])
                else:
                    return lambda X: (X['jet_features'], X['label'])
        else:
            var_index = tf.constant([int(i) for i in variables.split(",") if i])
            if parametric:
                if mass_ordering:
                    if weighted:
                        return lambda X: ((tf.gather(X['jet_features'], var_index, axis=1),
                                           X['param_masses_ordered']), X['label'], X['weight_merged'][0])
                    else:
                        return lambda X: ((tf.gather(X['jet_features'], var_index, axis=1),
                                           X['param_masses_ordered']), X['label'])
                else:
                    if weighted:
                        return lambda X: ((tf.gather(X['jet_features'], var_index, axis=1),
                                           X['param_masses_unordered']),
                                          X['label'], X['weight'][0])
                    else:
                        return lambda X: ((tf.gather(X['jet_features'], var_index, axis=1),
                                           X['param_masses_unordered']), X['label'])
            else:
                if weighted:
                    return lambda X: (tf.gather(X['jet_features'], var_index, axis=1),
                                      X['label'], X['weight'][0])
                else:
                    return lambda X: (tf.gather(X['jet_features'], var_index, axis=1),
                                      X['label'])
    else:
        if parametric:
            if mass_ordering:
                if weighted:
                    return lambda X: ((X['part_coords'], X['part_features'],
                                       X['part_masks'], X['jet_features'],
                                       X['param_masses_ordered']), X['label'], X['weight_merged'][0])
                else:
                    return lambda X: ((X['part_coords'], X['part_features'],
                                       X['part_masks'], X['jet_features'],
                                       X['param_masses_ordered']), X['label'])
            else:
                if weighted:
                    return lambda X: ((X['part_coords'], X['part_features'],
                                       X['part_masks'], X['jet_features'],
                                       X['param_masses_unordered']), X['label'], X['weight'][0])
                else:
                    return lambda X: ((X['part_coords'], X['part_features'],
                                       X['part_masks'], X['jet_features'],
                                       X['param_masses_unordered']), X['label'])
        else:
            if weighted:
                return lambda X: ((X['part_coords'], X['part_features'],
                                   X['part_masks'], X['jet_features']),
                                  X['label'], X['weight'][0])
            else:
                return lambda X: ((X['part_coords'], X['part_features'],
                                   X['part_masks'], X['jet_features']), X['label'])

def get_filter_fn(masses, mass_ordering:bool=True, mode:str="include",
                  ignore_bkg:bool=True):
    import tensorflow as tf
    masses_tensor = tf.constant(masses, dtype='float64')
    if mass_ordering:
        param_name = 'param_masses_ordered'
    else:
        param_name = 'param_masses_unordered'
    if mode == "exclude":
        if ignore_bkg:
            def filter_fn(x):
                return (x['label'][0] == 0) or tf.reduce_all(tf.reduce_any(tf.not_equal(x[param_name], masses_tensor), axis=1))
        else:
            def filter_fn(x):
                return tf.reduce_all(tf.reduce_any(tf.not_equal(x[param_name], masses_tensor), axis=1))
    elif mode == "include":
        if ignore_bkg:
            def filter_fn(x):
                return (x['label'][0] == 0) or tf.reduce_all(tf.reduce_all(tf.equal(x[param_name], masses_tensor), axis=1))
        else:
            def filter_fn(x):
                return tf.reduce_all(tf.reduce_all(tf.equal(x[param_name], masses_tensor), axis=1))
    else:
        raise ValueError(f"unknwon filter mode: {mode}")
    return filter_fn

def get_dataset(filenames, batch_size:int, parse_tfrecord_fn,
                input_fn, include_fn=None, exclude_fn=None,
                cache:bool=False, shuffle:bool=False,
                seed:int=2023):
    import tensorflow as tf
    from aliad.interface.tensorflow.dataset import apply_pipelines
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if include_fn is not None:
        ds = ds.filter(include_fn)
    if exclude_fn is not None:
        ds = ds.filter(exclude_fn)
    ds = ds.map(input_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # datasets pre-shuffled, no need to shuffle again
    if shuffle:
        metadata_filenames = [f"{os.path.splitext(fname)[0]}_metadata.json" for fname in filenames]
        buffer_size = np.sum([json.load(open(fname))['size'] for fname in metadata_filenames])
    else:
        buffer_size = None
    ds = apply_pipelines(ds, batch_size=batch_size,
                         cache=cache, prefetch=True,
                         shuffle=shuffle, seed=seed,
                         buffer_size=buffer_size,
                         drop_remainder=True)
    return ds

def get_ws_dataset(m1, m2, mu, high_level:bool,
                   dataset_dir:str, 
                   dataset_indices:Dict,
                   variables=None):
    all_ds = {}
    get_zero_label = get_input_fn(0, high_level=high_level, variables=variables)
    get_one_label = get_input_fn(1, high_level=high_level, variables=variables)
    batch_size = 1024 if high_level else 32
    metadata_filename = os.path.join(dataset_dir, 'QCD', f'SR_point_cloud_train_features_shuffled_shard_0_metadata.json')
    metadata = json.load(open(metadata_filename))
    parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'], downcast=True)
    for ds_type in ['train', 'val', 'test']:
        indices = dataset_indices[ds_type]
        filenames = {"sig": [], "bkg": []}
        sample_sizes = {"sig": 0, "bkg": 0}
        for key, label in [('sig', f'W_{m1}_{m2}'),
                           ('bkg', 'QCD')]:
            for index in indices:
                sample_filename = os.path.join(dataset_dir, label, f'SR_point_cloud_train_features_shuffled_shard_{index}.tfrec')
                metadata_filename = os.path.join(dataset_dir, label, f'SR_point_cloud_train_features_shuffled_shard_{index}_metadata.json')
                metadata = json.load(open(metadata_filename))
                sample_sizes[key] += metadata['size']
                filenames[key].append(sample_filename)
        sample_ds = {}
        for key in filenames:
            ds = tf.data.TFRecordDataset(filenames[key], num_parallel_reads=tf.data.AUTOTUNE)
            ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            sample_ds[key] = ds
        print(f"INFO: Creating {ds_type} dataset")
        nsig, nbkg = sample_sizes['sig'], sample_sizes['bkg']
        if ds_type != 'test':
            eff_nsig = int(mu * nbkg)
            eff_nbkg_0 = int(nbkg * 0.5)
            eff_nbkg_1 = nbkg - int(nbkg * 0.5)
            if eff_nsig > nsig:
                eff_nsig = nsig
                eff_nbkg_0  = int(nsig / (2 * mu))
                eff_nbkg_1  = int(nsig / (2 * mu))
            # half of bkg is labeled zero while the other half is labeled one
            #ds_bkg_0 = ds_bkg.take(eff_nbkg_1 + eff_nbkg_2).map(get_zero_label)
            ds_bkg_0 = sample_ds['bkg'].take(eff_nbkg_0).map(get_zero_label)
            ds_bkg_1 = sample_ds['bkg'].skip(eff_nbkg_0).take(eff_nbkg_1).map(get_one_label)
            ds_sig_1 = sample_ds['sig'].take(eff_nsig).map(get_one_label)
            eff_nbkg = eff_nbkg_0 + eff_nbkg_1
            print(f"Number of bkg = {eff_nbkg} (0-labeled: {eff_nbkg_0}, 1-labeled: {eff_nbkg_1}), Number of sig = {eff_nsig}")
            ds = ds_bkg_0.concatenate(ds_bkg_1).concatenate(ds_sig_1)
        else:
            ds = sample_ds['bkg'].map(get_zero_label).concatenate(sample_ds['sig'].map(get_one_label))
            print(f"Number of bkg = {nbkg}, Number of sig = {nsig}")
        if ds_type == 'train':
            shuffle = True
            buffer_size = eff_nbkg_0 + eff_nbkg_1 + eff_nsig
        else:
            shuffle = False
            buffer_size = None
        ds = apply_pipelines(ds, batch_size=batch_size,
                             shuffle=shuffle,
                             buffer_size=buffer_size,
                             drop_remainder=False)
        all_ds[ds_type] = ds
    return all_ds

# method to extract the feature vectors and label from the tfrecord
def get_ws_input_fn(label:int, high_level:bool=True, variables=None):
    import tensorflow as tf
    if high_level:
        if variables is None:
            if label == 0:
                return lambda X: (X['jet_features'], tf.zeros((1,), dtype='int64'))
            else:
                return lambda X: (X['jet_features'], tf.ones((1,), dtype='int64'))
        else:
            var_index = tf.constant([int(i) for i in variables.split(",")])
            if label == 0:
                return lambda X: (tf.gather(X['jet_features'], var_index, axis=1), tf.zeros((1,), dtype='int64'))
            else:
                return lambda X: (tf.gather(X['jet_features'], var_index, axis=1), tf.ones((1,), dtype='int64'))
    else:
        if label == 0:
            return lambda X: ((X['part_coords'], X['part_features'],
                               X['part_masks'], X['jet_features']),
                              tf.zeros((1,), dtype='int64'))
        else:
            return lambda X: ((X['part_coords'], X['part_features'],
                   X['part_masks'], X['jet_features']),
                  tf.ones((1,), dtype='int64'))
            
def get_ws_dataset(input_paths,
                   m1, m2, mu,
                   high_level:bool,
                   batch_size:int,
                   variables:str=None):
    import tensorflow as tf
    from aliad.interface.tensorflow.dataset import get_ndarray_tfrecord_example_parser, apply_pipelines
    m1, m2 = int(m1), int(m2)
    all_ds = {}
    get_zero_label = get_ws_input_fn(0, high_level=high_level, variables=variables)
    get_one_label = get_ws_input_fn(1, high_level=high_level, variables=variables)
    dataset_paths = input_paths['dataset']
    metadata_paths = input_paths['metadata']
    
    metadata_filename = os.path.join(dataset_dir, 'QCD', f'SR_point_cloud_train_features_shuffled_shard_0_metadata.json')
    metadata = json.load(open(metadata_filename))
    parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'], downcast=True)
    for ds_type in ['train', 'val', 'test']:
        indices = dataset_indices[ds_type]
        filenames = {"sig": [], "bkg": []}
        sample_sizes = {"sig": 0, "bkg": 0}
        for key, label in [('sig', f'W_{m1}_{m2}'),
                           ('bkg', 'QCD')]:
            for index in indices:
                sample_filename = os.path.join(dataset_dir, label, f'SR_point_cloud_train_features_shuffled_shard_{index}.tfrec')
                metadata_filename = os.path.join(dataset_dir, label, f'SR_point_cloud_train_features_shuffled_shard_{index}_metadata.json')
                metadata = json.load(open(metadata_filename))
                sample_sizes[key] += metadata['size']
                filenames[key].append(sample_filename)
        sample_ds = {}
        for key in filenames:
            ds = tf.data.TFRecordDataset(filenames[key], num_parallel_reads=tf.data.AUTOTUNE)
            ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            sample_ds[key] = ds
        print(f"INFO: Creating {ds_type} dataset")
        nsig, nbkg = sample_sizes['sig'], sample_sizes['bkg']
        if ds_type != 'test':
            eff_nsig = int(mu * nbkg)
            eff_nbkg_0 = int(nbkg * 0.5)
            eff_nbkg_1 = nbkg - int(nbkg * 0.5)
            if eff_nsig > nsig:
                eff_nsig = nsig
                eff_nbkg_0  = int(nsig / (2 * mu))
                eff_nbkg_1  = int(nsig / (2 * mu))
            # half of bkg is labeled zero while the other half is labeled one
            #ds_bkg_0 = ds_bkg.take(eff_nbkg_1 + eff_nbkg_2).map(get_zero_label)
            ds_bkg_0 = sample_ds['bkg'].take(eff_nbkg_0).map(get_zero_label)
            ds_bkg_1 = sample_ds['bkg'].skip(eff_nbkg_0).take(eff_nbkg_1).map(get_one_label)
            ds_sig_1 = sample_ds['sig'].take(eff_nsig).map(get_one_label)
            eff_nbkg = eff_nbkg_0 + eff_nbkg_1
            print(f"Number of bkg = {eff_nbkg} (0-labeled: {eff_nbkg_0}, 1-labeled: {eff_nbkg_1}), Number of sig = {eff_nsig}")
            ds = ds_bkg_0.concatenate(ds_bkg_1).concatenate(ds_sig_1)
        else:
            ds = sample_ds['bkg'].map(get_zero_label).concatenate(sample_ds['sig'].map(get_one_label))
            print(f"Number of bkg = {nbkg}, Number of sig = {nsig}")
        if ds_type == 'train':
            shuffle = True
            buffer_size = eff_nbkg_0 + eff_nbkg_1 + eff_nsig
        else:
            shuffle = False
            buffer_size = None
        ds = apply_pipelines(ds, batch_size=batch_size,
                             shuffle=shuffle,
                             buffer_size=buffer_size,
                             drop_remainder=False)
        all_ds[ds_type] = ds
    return all_ds


def get_train_config(checkpoint_dir:str, high_level:bool=True):
    
    config = {
        # for binary classification
        'loss'       : 'binary_crossentropy',
        'metrics'    : ['accuracy'],
        'epochs'     : 100 if high_level else 20,
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
                'patience': 10 if high_level else 5
            },
            'model_checkpoint':{
                'save_weights_only': True,
                # save model checkpoint every epoch
                'save_freq': 'epoch'
            },
            'metrics_logger':{
                'save_freq': -1
            }
        }
    }

    return config

def get_required_features(high_level:bool, parametric:bool,
                          mass_ordering:bool=False,
                          weighted:bool=False):
    if high_level:
        features = ['jet_features', 'label']
    else:
        features = ['part_coords', 'part_features',
                    'part_masks', 'jet_features', 'label']
    if parametric:
        if mass_ordering:
            features.append('param_masses_ordered')
            if weighted:
                features.append('weight_merged')
        else:
            features.append('param_masses_unordered')
            if weighted:
                features.append('weight')
    return features
    
def get_dedicated_sample_paths(dirname:str, mass_point, decay_modes:List[str], extra_bkg:bool=True):
    m1, m2 = mass_point
    samples = [f'QCD_qq_{m1}_{m2}']
    if extra_bkg:
        samples.append(f'extra_QCD_qq_{m1}_{m2}')
    for decay_mode in decay_modes:
        samples.append(f'W_{decay_mode}_{m1}_{m2}')
    paths = {
        "dataset"  : {},
        "metadata" : {}
    }
    for sample in samples:
        sample_dir = os.path.join(dirname, sample)
        if not os.path.exists(sample_dir):
            raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")
        ds_filenames = glob.glob(os.path.join(sample_dir, "*.tfrec"))
        if not ds_filenames:
            raise RuntimeError(f"no tfrecord datasets found for the sample '{sample}' "
                               f"under the directory '{sample_dir}'")
        ds_filenames = sorted(ds_filenames, key = lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
        metadata_filenames = [f"{os.path.splitext(fname)[0]}_metadata.json" for fname in ds_filenames]
        paths['dataset'][sample]  = ds_filenames
        paths['metadata'][sample] = metadata_filenames
    return paths

def get_ds_split_specs(paths, split_config:Dict):
    dataset_paths = paths['dataset']
    metadata_paths = paths['metadata']
    stages = list(split_config)
    specs = {}
    def _get_sample_size(metadata_path):
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        return metadata['size']
    # dedicated samples
    if isinstance(dataset_paths, dict):
        samples = list(dataset_paths)
        for stage in stages:
            indices = split_config[stage]
            specs[stage] = {
                'path'   : [],
                'sample' : [],
                'size'   : []
            }
            for sample in samples:
                specs[stage]['path'].extend([dataset_paths[sample][index] for index in indices])
                specs[stage]['sample'].extend([sample] * len(indices))
                specs[stage]['size'].extend([_get_sample_size(metadata_paths[sample][index]) for index in indices])
    # mixed samples
    else:
        for stage in stages:
            specs[stage] = {
                'path'   : [dataset_paths[index] for index in indices],
                'sample' : [None] * len(indices),
                'size'   : [_get_sample_size(metadata_paths[index]) for index in indices]
            }
    return specs 
    
def get_callbacks(config:Dict, semi_weakly:bool=False):
    
    from aliad.interface.tensorflow.callbacks import LearningRateScheduler, MetricsLogger, WeightsLogger
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    checkpoint_dir = config['checkpoint_dir']
    
    lr_scheduler = LearningRateScheduler(**config['callbacks']['lr_scheduler'])
    
    early_stopping = EarlyStopping(**config['callbacks']['early_stopping'])
    
    checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'model_weights_epoch_{epoch:02d}.h5'),
                                 **config['callbacks']['model_checkpoint'])
    metrics_logger = MetricsLogger(checkpoint_dir, **config['callbacks']['metrics_logger'])

    callbacks = [lr_scheduler, early_stopping, checkpoint, metrics_logger]

    if semi_weakly:
        weights_logger = WeightsLogger(checkpoint_dir, **config['callbacks']['weights_logger'])
        callbacks.append(weights_logger)

    return callbacks

def suggest_batchsize(batchsize:Optional[int]=None, high_level:bool=True):
    if batchsize is None:
        batchsize = 1024 if high_level else 128
    return batchsize

def compile_model(model, config:Dict):
    import tensorflow as tf
    optimizer = tf.keras.optimizers.get({'class_name': config['optimizer'],
                                         'config': config['optimizer_config']})
    model.compile(loss=config['loss'],
                  optimizer=optimizer,
                  metrics=config['metrics'])
    
def load_model(model_path:str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)
    
def freeze_all_layers(model):
    # equivalent to freeze_model
    for layer in model.layers:
        layer.trainable = False
        
def freeze_model(model):
    model.trainable = False

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


def get_ws_weight_models(m1:float, m2:float, mu:Optional[float]=None):
    import tensorflow as tf
    weight_models = {
        'm1': get_single_parameter_model(kernel_initializer=tf.constant_initializer(m1)),
        'm2': get_single_parameter_model(kernel_initializer=tf.constant_initializer(m2)),
    }
    if mu is not None:
        weight_models['mu'] = get_single_parameter_model(activation='linear',
                                                         exponential=True,
                                                         kernel_initializer=tf.constant_initializer(mu))        
    return weight_models  

def get_ws_model(feature_inputs, inputs, mu, fs_model):
    from tensorflow.keras import Model
    fs_out = fs_model(inputs)
    LLR = fs_out / (1. - fs_out)
    LLR_xs = 1. + mu * (LLR - 1.)
    ws_out = LLR_xs / (1 + LLR_xs)
    model = Model(inputs=feature_inputs, outputs=ws_out)
    return model