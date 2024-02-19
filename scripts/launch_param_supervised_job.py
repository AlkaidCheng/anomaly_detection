#! /usr/bin/env python
from typing import Optional

import click

from utils import (get_high_level_model, get_low_level_model)
from utils import (get_input_fn, get_filter_fn)
from utils import (get_train_config, get_ds_split_specs,
                   get_dataset, get_required_features,
                   suggest_batchsize)

def run_param_supervised(high_level:bool=True, decay_mode:str='qq',
                         weighted:bool=True, mass_ordering:bool=False,
                         variables:Optional[str]=None, 
                         include_masses:Optional[str]=None,
                         exclude_masses:Optional[str]=None,
                         interrupt_freq:Optional[int]=None,
                         class_weight_ratio:float=1.0,
                         cycle_bkg_param:bool=True,
                         dataset_index_file:str=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                                 "LHCO_RnD/tfrecords/dataset_indices.json"),
                         split_index:int=0,
                         batchsize:Optional[int]=None,
                         dirname:str=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                      "LHCO_RnD/tfrecords/shuffled_{feature_level}"),
                         outdir:str="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD_new/fully_supervised/RnD_{decay_mode}",
                         cache_dataset:Optional[bool]=None,
                         version_str:str='v1', cache:bool=True, multi_gpu:bool=False):
    import os
    import glob
    import json
    import time
    
    import numpy as np
    import tensorflow as tf
    import aliad as ad

    from quickstats.utils.common_utils import NpEncoder
    from aliad.interface.tensorflow.dataset import (apply_pipelines, split_dataset,
                                                    get_ndarray_tfrecord_example_parser)
    from aliad.interface.tensorflow.callbacks import LearningRateScheduler, MetricsLogger, EarlyStopping
    from tensorflow.keras.callbacks import ModelCheckpoint

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

    filenames = glob.glob(os.path.join(dirname, '*.tfrec'))
    # sort by the shard index
    filenames = sorted(filenames, key = lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
    metadata_filenames = glob.glob(os.path.join(dirname, '*metadata.json'))
    
    # metadata contains information about the shape and dtype of each type of features, as well as the size of the dataset
    metadata = json.load(open(metadata_filenames[-1]))

    if high_level:
        required_keys = ['jet_features', 'label']
    else:
        required_keys = ['part_coords', 'part_features',
                         'part_masks', 'jet_features', 'label']

    if mass_ordering:
        required_keys.append('param_masses_ordered')
        if weighted:
            required_keys.append('weight_merged')
    else:
        required_keys.append('param_masses_unordered')
        if weighted:
            required_keys.append('weight')

    # method for parsing the binary tfrecord into array data
    parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'])
    
    batchsize = suggest_batchsize(batchsize, high_level=high_level)
    input_fn = get_input_fn(high_level=high_level,
                            parametric=True,
                            mass_ordering=mass_ordering,
                            weighted=weighted,
                            variables=variables)
    with open(index_file, 'r') as file:
        ds_split_config = json.load(file)
    ds_splits = ds_split_config[str(split_index)]
    
    if exclude_masses is not None:
        exclude_masses = [m for m in exclude_masses.split(",") if m]
        exclude_masses = [[int(j) for j in k.split(":") if j] for k in exclude_masses]
        exclude_fn = get_filter_fn(exclude_masses, mass_ordering=mass_ordering, mode="exclude", ignore_bkg=False)
    else:
        exclude_fn = None

    if include_masses is not None:
        include_masses = [m for m in include_masses.split(",") if m]
        include_masses = [[int(j) for j in k.split(":") if j] for k in include_masses]
        include_fn = get_filter_fn(include_masses, mass_ordering=mass_ordering, mode="include",
                                   ignore_bkg=not cycle_bkg_param)
    else:
        include_fn = None

    all_ds = {}
    if cache_dataset is None:
        cache_dataset = high_level
    for stage in ds_splits:
        stage_filenames = [filenames[i] for i in ds_splits[stage]]
        all_ds[stage] = get_dataset(stage_filenames, batch_size=batchsize,
                                    parse_tfrecord_fn=parse_tfrecord_fn,
                                    input_fn=input_fn,
                                    include_fn=include_fn,
                                    exclude_fn=exclude_fn,
                                    cache=cache_dataset and stage != 'test')
    checkpoint_dir = "high_level" if high_level else "low_level"
    checkpoint_dir = os.path.join(checkpoint_dir, "parameterised")
    checkpoint_dir = os.path.join(checkpoint_dir, "mass_ordered" if mass_ordering else "mass_unordered")
    checkpoint_dir = os.path.join(checkpoint_dir, "weighted" if weighted else "unweighted")
    basename = f"SR_10M_events_ratio_{int(class_weight_ratio)}_"
    basename += 'var_' + '_'.join(variables.split(",")) + '_' if (variables) else 'var_all_'
    basename += version_str
    checkpoint_dir = os.path.join(outdir, checkpoint_dir, basename, f"split_{split_index}")

    config = {
        # for binary classification
        'loss'       : 'binary_crossentropy',
        'metrics'    : ['accuracy'],
        'epochs'     : 100 if high_level else 10,
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
                'patience': 10 if high_level else 3,
                'interrupt_freq': interrupt_freq,
                'restore_best_weights': True
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

    os.makedirs(checkpoint_dir, exist_ok=True)

    if high_level:
        model_fn = get_high_level_model
    else:
        model_fn = get_low_level_model

    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = model_fn(metadata['features'],
                             parametric=True,
                             mass_ordering=mass_ordering,
                             variables=variables)
            optimizer = tf.keras.optimizers.get({'class_name': config['optimizer'],
                                                 'config': config['optimizer_config']})
            model.compile(loss=config['loss'],
                          optimizer=optimizer,
                          metrics=config['metrics'])
    else:
        model = model_fn(metadata['features'],
                         parametric=True,
                         mass_ordering=mass_ordering,
                         variables=variables)
        optimizer = tf.keras.optimizers.get({'class_name': config['optimizer'],
                                             'config': config['optimizer_config']})
        model.compile(loss=config['loss'],
                      optimizer=optimizer,
                      metrics=config['metrics'])
        
    lr_scheduler = LearningRateScheduler(**config['callbacks']['lr_scheduler'])
    
    early_stopping = EarlyStopping(**config['callbacks']['early_stopping'])
    
    metrics_ckpt_filepath = os.path.join(checkpoint_dir, "epoch_metrics",
                                         "metrics_epoch_{epoch}.json")
    model_ckpt_filepath = os.path.join(checkpoint_dir, "model_weights_epoch_{epoch:02d}.h5")

    if interrupt_freq:
        early_stopping.restore(model, metrics_ckpt_filepath=metrics_ckpt_filepath,
                               model_ckpt_filepath=model_ckpt_filepath)

    checkpoint = ModelCheckpoint(model_ckpt_filepath,
                                 **config['callbacks']['model_checkpoint'])
    metrics_logger = MetricsLogger(checkpoint_dir, **config['callbacks']['metrics_logger'])
    callbacks = [lr_scheduler, early_stopping, checkpoint, metrics_logger]
    
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
                  class_weight=class_weight,
                  initial_epoch=early_stopping.initial_epoch)
        if early_stopping.interrupted:
            print("INFO: Training interrupted!")
            return None
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
        if weighted:
            y_true = np.concatenate([y for _, y, _ in all_ds['test']]).flatten()
        else:
            y_true = np.concatenate([y for _, y in all_ds['test']]).flatten()
                
        print("INFO: Finished prediction!")
        results = {
            'predicted_proba': predicted_proba,
            'y_true': y_true
        }
        if weighted:
            masses = np.concatenate([x[-1] for x, _, _ in all_ds['test']])
        else:
            masses = np.concatenate([x[-1] for x, _ in all_ds['test']])
        masses = masses.reshape([-1, 2])
        results['m1'] = masses[:, 0]
        results['m2'] = masses[:, 1]
        with open(result_filename, 'w') as f:
            json.dump(results, f, cls=NpEncoder)
    t3 = time.time()
    print(f'INFO: Test time = {t3 - t2:.3f}s')
    
    print(f"INFO: Job finished! Total time taken = {t3 - t0:.3f}s")

    

@click.command(name='run')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='whether to do training with low-evel or high-level features.')
@click.option('--weighted/--unweighted', default=True, show_default=True,
              help='whether to use sample weights in training')
@click.option('--class-weight-ratio', type=float, default=1.0, show_default=True,
              help='ratio between the class weights of background to signal')
@click.option('--mass-ordering/--no-mass-ordering', default=False, show_default=True,
              help='whether to order the parametric masses')
@click.option('--cycle-bkg-param/--random-bkg-param', default=True, show_default=True,
              help='whether the backgrounds are parameterised by cycling data or random assignment')
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
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='batch size')
@click.option('--interrupt-freq', default=None, type=int, show_default=True,
              help='How many more epochs to train before interrupting')
@click.option('--decay-mode', default='qq', type=click.Choice(['qq', 'qqq'], case_sensitive=False), show_default=True,
              help='which decay mode should the signal undergo (qq or qqq)')
@click.option('--exclude-masses', default=None, show_default=True,
              help='mass points to exclude (mass point separated by commas, mass values separated by colon)')
@click.option('--include-masses', default=None, show_default=True,
              help='mass points to include (mass point separated by commas, mass values separated by colon)')
@click.option('--dirname',
              default=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                       "LHCO_RnD/tfrecords/shuffled_{feature_level}"),
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
              help='whether to use multi-GPU for training')
def cli(**kwargs):
    run_param_supervised(**kwargs)

if __name__ == "__main__":
    cli()