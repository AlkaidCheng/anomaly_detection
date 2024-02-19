import os
import glob
import json

import numpy as np
import tensorflow as tf

from aliad.interface.tensorflow.dataset import (apply_pipelines, split_dataset,
                                                get_ndarray_tfrecord_example_parser)

dirname = '/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/LHCO_RnD_qq/tfrecords/shuffled'
filenames = glob.glob(os.path.join(dirname, '*.tfrec'))
# sort by the shard index
filenames = sorted(filenames, key = lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
metadata_filenames = glob.glob(os.path.join(dirname, '*metadata.json'))

# metadata contains information about the shape and dtype of each type of features, as well as the size of the dataset
metadata = json.load(open(metadata_filenames[-1]))


# method for parsing the binary tfrecord into array data
parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'])

# method to extract the feature vectors and label from the tfrecord
def get_input_fn(high_level:bool=True, parametric:bool=True, mass_ordering:str=False, weighted:bool=True):
    if high_level:
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

def get_dataset(filenames, batch_size:int, input_fn):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(input_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # datasets pre-shuffled, no need to shuffle again
    ds = apply_pipelines(ds, batch_size=batch_size,
                         shuffle=False, prefetch=True)
    return ds

def get_high_level_model(feature_metadata, parametric:bool=True, mass_ordering:bool=True):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dense, Flatten
    import tensorflow as tf
    if parametric:
        label = 'param_masses_ordered' if mass_ordering else 'param_masses_unordered'
        x1 = Input(**feature_metadata['jet_features'])
        x2 = Input(**feature_metadata[label])
        inputs = [x1, x2]
        x  = tf.concat([x1, tf.expand_dims(x2, axis=-1)], -1)
        x = tf.reshape(x, (-1, tf.reduce_prod(tf.shape(x)[1:])))
    else:
        x1 = Input(**feature_metadata['jet_features'])
        inputs = x1
        x = x1
    layers = [(256, 'relu'),
              (128, 'relu'),
              (64, 'relu'),
              (1, 'sigmoid')]
    for nodes, activation in layers:
        x = Dense(nodes, activation)(x)
    model = Model(inputs=inputs, outputs=x, name='HighLevel')
    return model

def get_low_level_model(feature_metadata, parametric:bool=True, mass_ordering:bool=True):
    from tensorflow.keras.layers import Input
    from aliad.interface.tensorflow.models import ModifiedParticleNet
    inputs = {
        'points': Input(shape=feature_metadata['part_coords']['shape']),
        'features': Input(shape=feature_metadata['part_features']['shape']),
        'masks': Input(**feature_metadata['part_masks']),
        'jet_features': Input(shape=feature_metadata['jet_features']['shape'])
    }
    if parametric:
        label = 'param_masses_ordered' if mass_ordering else 'param_masses_unordered'
        inputs['param_features'] = Input(shape=feature_metadata[label]['shape'])
    model_builder = ModifiedParticleNet()
    model = model_builder.get_model(**inputs)
    return model

from aliad.interface.tensorflow.callbacks import LearningRateScheduler, MetricsLogger

import os
import json
from quickstats.utils.common_utils import NpEncoder
basedir = "/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD/fully_supervised/"
scenarios = ["param_low_level_10M_events_SR_mass_ordered_unweighted_ratio_1_v1",
             "param_low_level_10M_events_SR_mass_ordered_weighted_ratio_2_v1"]
for scenario in scenarios:
    print(scenario)
    high_level = 'high_level' in scenario
    batch_size = 1024 if high_level else 32
    parametric = scenario.startswith('param')
    mass_ordering = 'mass_ordered' in scenario
    weighted = 'unweighted' not in scenario
    checkpoint_dir = os.path.join(basedir, scenario)
    config = {
        # for binary classification
        'loss'       : 'binary_crossentropy',
        'metrics'    : ['accuracy'],
        'epochs'     : 100,
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
                'patience': 3
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
    config['checkpoint_dir'] = checkpoint_dir
    model_path = os.path.join(checkpoint_dir, "full_train.keras")
    if not os.path.exists(model_path):
        if high_level:
            model_fn = get_high_level_model
        else:
            model_fn = get_low_level_model
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = model_fn(metadata['features'],
                             parametric=parametric,
                             mass_ordering=mass_ordering)
        metrics_logger = MetricsLogger(checkpoint_dir, **config['callbacks']['metrics_logger'])
        metrics_logger.restore()
        df = metrics_logger.get_dataframe('epoch')
        best_epoch = int(df.iloc[df['val_loss'].argmin()]['epoch'])
        weight_path = os.path.join(checkpoint_dir, f"model_weights_epoch_{best_epoch+1:02}.h5")
        print(weight_path)
        model.load_weights(weight_path)

        model.save(model_path)
    else:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(model_path)
    result_filename = os.path.join(config['checkpoint_dir'], 'test_results.json')
    if os.path.exists(result_filename):
        continue
    results = json.load(open(result_filename))
    class_weight_ratio = 1.0
    # 50% Train, 25% Validation, 25% Test
    splits_indices = [[0, 50], [50, 75], [75, 100]]

    input_fn = get_input_fn(high_level=high_level,
                            parametric=parametric,
                            mass_ordering=mass_ordering,
                            weighted=weighted)
    all_ds = {}
    for index_range, stage in zip(splits_indices, ['train', 'val', 'test']):
        start, end = index_range
        all_ds[stage] = get_dataset(filenames[start:end], batch_size=batch_size, input_fn=input_fn)
    
    predicted_proba = model.predict(all_ds['test']).flatten()
    weights = np.ones(len(results['y_true']))
    if weighted:
        y_true = np.array([y for _, y, _ in all_ds['test']]).flatten()
        weights = np.array([w for _, _, w in all_ds['test']]).flatten()
    else:
        y_true = np.array([y for _, y in all_ds['test']]).flatten()

    with open(result_filename, 'w') as f:
        json.dump(results, f, cls=NpEncoder)
    print("INFO: Finished prediction!")
    results = {
        'predicted_proba': predicted_proba,
        'y_true': y_true
    }
    if parametric:
        if weighted:
            masses = np.array([x[-1] for x, _, _ in all_ds['test']])
        else:
            masses = np.array([x[-1] for x, _ in all_ds['test']])
        masses = masses.reshape([-1, 2])
        results['m1'] = masses[:, 0]
        results['m2'] = masses[:, 1]
    results['weight'] = weights
    with open(result_filename, 'w') as f:
        json.dump(results, f, cls=NpEncoder)