#! /usr/bin/env python
from typing import Optional

import click

from utils import (get_high_level_model, get_low_level_model)
from utils import (get_input_fn, get_filter_fn)
from utils import get_dataset, get_dedicated_sample_paths, get_ds_split_specs
from utils import get_train_config, get_required_features, suggest_batchsize

def run_dedicated_supervised(mass_point:str,
                             high_level:bool=True, decay_mode:str='qq',
                             variables:Optional[str]=None,
                             class_weight_ratio:float=1.0,
                             dataset_index_file:str=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                                     "LHCO_RnD/tfrecords/dataset_indices.json"),
                             split_index:int=0,
                             seed:int=2023,
                             batchsize:Optional[int]=None,
                             dirname:str=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                          "LHCO_RnD/tfrecords/samples_{feature_level}"),
                             outdir:str="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD_new/fully_supervised/RnD_{decay_mode}",
                             cache_dataset:Optional[bool]=None,
                             version_str:str='v1', cache:bool=True, multi_gpu:bool=False):
    import os
    import glob
    import json
    import time
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import aliad as ad

    from quickstats.utils.common_utils import NpEncoder
    from quickstats.utils.string_utils import split_str
    from aliad.interface.tensorflow.dataset import (apply_pipelines, split_dataset,
                                                    get_ndarray_tfrecord_example_parser)
    from aliad.interface.tensorflow.callbacks import LearningRateScheduler, MetricsLogger
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    t0 = time.time()
    feature_level = "high_level" if high_level else "low_level"
    dirname = dirname.format(decay_mode=decay_mode, feature_level=feature_level)
    outdir = outdir.format(decay_mode=decay_mode)
    index_file = dataset_index_file.format(decay_mode=decay_mode)
    
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

    
    m1, m2 = split_str(mass_point, sep=':', remove_empty=True, strip=True)
    sample_paths = get_dedicated_sample_paths(dirname, mass_point=[m1, m2],
                                              decay_modes=[decay_mode])
    samples = list(sample_paths['dataset'])
    # metadata contains information about the shape and dtype of each type of features, as well as the size of the dataset
    metadata = json.load(open(sample_paths['metadata'][samples[-1]][-1]))

    required_features = get_required_features(high_level=high_level, parametric=False)
            
    # method for parsing the binary tfrecord into array data
    parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'], keys=required_features)

    batchsize = suggest_batchsize(batchsize, high_level=high_level)

    input_fn = get_input_fn(high_level=high_level,
                            parametric=False,
                            mass_ordering=False,
                            weighted=False,
                            variables=variables)
    
    with open(index_file, 'r') as file:
        ds_split_config = json.load(file)
    split_config = ds_split_config[str(split_index)]
    split_specs = get_ds_split_specs(sample_paths, split_config)

    print('Number of available events in different dataset splits:')
    print(pd.DataFrame(split_specs('size')))

    all_ds = {}
    if cache_dataset is None:
        cache_dataset = high_level
    for stage, filenames in split_specs['dataset'].items():
        all_ds[stage] = get_dataset(filenames, batch_size=batchsize,
                                    parse_tfrecord_fn=parse_tfrecord_fn,
                                    input_fn=input_fn,
                                    cache=cache_dataset and stage != "test",
                                    shuffle=stage=='train',
                                    seed=seed)

    checkpoint_dir = "high_level" if high_level else "low_level"
    checkpoint_dir = os.path.join(checkpoint_dir, "dedicated")
    checkpoint_dir = os.path.join(checkpoint_dir, f"W_{decay_mode}_{int(m1)}_{int(m2)}")
    basename = f"SR_10M_events_ratio_{int(class_weight_ratio)}_"
    basename += 'var_' + '_'.join(variables.split(",")) + '_' if (variables) else 'var_all_'
    basename += version_str
    checkpoint_dir = os.path.join(outdir, checkpoint_dir, basename, f"split_{split_index}")

    config = get_train_config(checkpoint_dir, high_level=high_level)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    lr_scheduler = LearningRateScheduler(**config['callbacks']['lr_scheduler'])
    
    early_stopping = EarlyStopping(**config['callbacks']['early_stopping'])
    
    checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'model_weights_epoch_{epoch:02d}.h5'),
                                 **config['callbacks']['model_checkpoint'])
    metrics_logger = MetricsLogger(checkpoint_dir, **config['callbacks']['metrics_logger'])
    callbacks = [lr_scheduler, early_stopping, checkpoint, metrics_logger]

    if high_level:
        model_fn = get_high_level_model
    else:
        model_fn = get_low_level_model

    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = model_fn(metadata['features'],
                             parametric=False,
                             mass_ordering=False,
                             variables=variables)
            optimizer = tf.keras.optimizers.get({'class_name': config['optimizer'],
                                                 'config': config['optimizer_config']})
            model.compile(loss=config['loss'],
                          optimizer=optimizer,
                          metrics=config['metrics'])
    else:
        model = model_fn(metadata['features'],
                         parametric=False,
                         mass_ordering=False,
                         variables=variables)
        optimizer = tf.keras.optimizers.get({'class_name': config['optimizer'],
                                             'config': config['optimizer_config']})
        model.compile(loss=config['loss'],
                      optimizer=optimizer,
                      metrics=config['metrics'])
    model_filename = os.path.join(config['checkpoint_dir'],
                                  "full_train.keras")
    if class_weight_ratio != 1.0:
        class_weight = {0: class_weight_ratio, 1: 1.0}
    else:
        class_weight = None

    print(f'INFO: Input directory = "{dirname}"')
    print(f'INFO: Output directory = "{outdir}"')
    print(f'INFO: Checkpoint directory = "{checkpoint_dir}"')

    t1 = time.time()

    print(f'INFO: Preparation time = {t1 - t0:.3f}s')
    # run model training
    if os.path.exists(model_filename) and cache:
        print(f'INFO: Cached model from "{model_filename}"')
        model = tf.keras.models.load_model(model_filename)
    else:
        model.fit(all_ds['train'],
                  validation_data=all_ds['val'],
                  epochs=config['epochs'],
                  callbacks=callbacks,
                  class_weight=class_weight)
        print("INFO: Finished training!")
        model.save(model_filename)
    t2 = time.time()
    print(f'INFO: Training time = {t2 - t1:.3f}s')
    # release memory
    for ds_type in all_ds:
        if ds_type != 'test':
            all_ds[ds_type] = None
    # run prediction
    result_filename = os.path.join(config['checkpoint_dir'], 'test_results.json')
    if os.path.exists(result_filename) and cache:
        print(f'INFO: Cached test results from "{result_filename}"!')
    else:
        predicted_proba = np.concatenate(model.predict(all_ds['test'])).flatten()
        y_true = np.concatenate([y for _, y in all_ds['test']]).flatten()
                
        print("INFO: Finished prediction!")
        results = {
            'predicted_proba': predicted_proba,
            'y_true': y_true
        }
        with open(result_filename, 'w') as f:
            json.dump(results, f, cls=NpEncoder)
    t3 = time.time()
    print(f'INFO: Test time = {t3 - t2:.3f}s')
    
    print(f"INFO: Job finished! Total time taken = {t3 - t0:.3f}s")

    

@click.command(name='run')
@click.option('--mass-point', required=True, show_default=True,
              help='signal mass point to use for training in the form "m1:m2"')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='whether to do training with low-evel or high-level features.')
@click.option('--class-weight-ratio', type=float, default=1.0, show_default=True,
              help='ratio between the class weights of background to signal')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='whether to cache the dataset during training')
@click.option('--variables', default=None, show_default=True,
              help='variable indices separated by commas to use in high-level training')
@click.option('--dataset-index-file', default="/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                              "LHCO_RnD/tfrecords/dataset_indices.json",
              show_default=True,
              help='config file with dataset split indices')
@click.option('--split-index', default=0, type=int, show_default=True,
              help='which dataset split index to use')
@click.option('--seed', default=2023, type=int, show_default=True,
              help='random seed for shuffling dataset')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='batch size')
@click.option('--decay-mode', default='qq', type=click.Choice(['qq', 'qqq'], case_sensitive=False), show_default=True,
              help='which decay mode should the signal undergo (qq or qqq)')
@click.option('--dirname',
              default=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                       "LHCO_RnD/tfrecords/samples_{feature_level}"),
              show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('--outdir',
              default="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD_new/fully_supervised/RnD_{decay_mode}",
              show_default=True,
              help='base output directory')
@click.option('--version', 'version_str',
              default="v1",
              show_default=True,
              help='version text')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='cache results when applicable')
@click.option('--multi-gpu/--single-gpu', default=False, show_default=True,
              help='whether to use multiple GPUs for training')
def cli(**kwargs):
    run_dedicated_supervised(**kwargs)

if __name__ == "__main__":
    cli()