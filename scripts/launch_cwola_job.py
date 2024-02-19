#! /usr/bin/env python
from typing import Optional

import click

from utils import (get_high_level_model, get_low_level_model)
from utils import (get_input_fn, get_filter_fn)
from utils import (get_train_config, get_ws_dataset, get_callbacks, compile_model)

def run_cwola(mass_point:str, mu:float, decay_mode:str='qq', high_level:bool=True, variables:Optional[str]=None,
              dataset_index_file:str=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                      "LHCO_RnD/tfrecords/dataset_indices.json"),
              split_index:int=0,
              dirname:str="/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                          "LHCO_RnD/tfrecords/samples_{feature_level}",
              outdir:str="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD_new/cwola/RnD_{decay_mode}",
              version_str:str='v1', cache:bool=True, multi_gpu:bool=False):
    import os
    import glob
    import json
    import time
    
    import numpy as np
    import tensorflow as tf
    import aliad as ad

    from quickstats.utils.common_utils import NpEncoder
    from quickstats.utils.string_utils import split_str
    from quickstats.maths.numerics import str_encode_value

    t0 = time.time()
    m1, m2 = split_str(mass_point, sep=":", remove_empty=True, cast=int)
    dirname = dirname.format(decay_mode=decay_mode)
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
        
    batch_size = 1024 if high_level else 32

    checkpoint_dir = "high_level" if high_level else "low_level"
    checkpoint_dir = os.path.join(checkpoint_dir, f"W_qq_{int(m1)}_{int(m2)}")
    basename = f"SR_10M_events_"
    basename += 'var_' + '_'.join(variables.split(",")) + '_' if (variables) else 'var_all_'
    basename += version_str
    mu_str = str_encode_value(mu)
    checkpoint_dir = os.path.join(outdir, checkpoint_dir, basename, f'mu_{mu_str}', f"split_{split_index}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = get_train_config(checkpoint_dir=checkpoint_dir, high_level=high_level)
    
    callbacks = get_callbacks(config)

    sample_paths = get_dedicated_sample_paths(dirname, mass_point=[m1, m2],
                                              decay_modes=[decay_mode])
    samples = list(sample_paths['dataset'])
    # metadata contains information about the shape and dtype of each type of features, as well as the size of the dataset
    metadata = json.load(open(sample_paths['metadata'][samples[-1]][-1]))

    with open(index_file, 'r') as file:
        ds_index_config = json.load(file)
    ds_indices = ds_index_config[str(split_index)]

    all_ds = get_ws_dataset(m1, m2, mu, high_level=high_level, batch_size=batch_size,
                            dataset_dir=dirname, dataset_indices=ds_indices,
                            variables=variables)
    
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
            compile_model(model, config)
    else:
        model = model_fn(metadata['features'],
                         parametric=parametric,
                         mass_ordering=mass_ordering,
                         variables=variables)
        compile_model(model, config)
        
    model_filename = os.path.join(config['checkpoint_dir'],
                                  "full_train.keras")

    print(f'##############################################################')
    print(f'INFO: CWoLa training with (m1, m2, mu) = ({m1}, {m2}, {mu}) in decay mode {decay_mode}')
    feature_level_txt = 'high level' if high_level else 'low_level'
    print(f'INFO: Training features = {feature_level_txt}')
    if high_level:
        var_txt = 'all' if variables is None else variables
        print(f'INFO: Feature indices = {var_txt}')
    print(f'INFO: Input directory = "{dirname}"')
    print(f'INFO: Output directory = "{outdir}"')
    print(f'INFO: Checkpoint directory = "{checkpoint_dir}"')
    print(f'##############################################################')

    # run model training
    if os.path.exists(model_filename) and cache:
        print(f'INFO: Cached model from "{model_filename}"')
        model = tf.keras.models.load_model(model_filename)
    else:
        model.fit(all_ds['train'],
                  validation_data=all_ds['val'],
                  epochs=config['epochs'],
                  callbacks=callbacks)
        print("INFO: Finished training!")
        model.save(model_filename)

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
    print("INFO: Job finished!")


@click.command(name='run')
@click.option('--mass-point', required=True,
              help='signal mass point to train for (e.g. "500:100")')
@click.option('--mu', required=True, type=float,
              help='signal fraction')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='whether to do training with low-evel or high-level features.')
@click.option('--variables', default=None, show_default=True,
              help='variable indices separated by commas to use in high-level training')
@click.option('--dataset-index-file', default=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                               "LHCO_RnD/tfrecords/dataset_indices.json"),
              show_default=True,
              help='config file with dataset split indices')
@click.option('--split-index', default=0, type=int, show_default=True,
              help='which dataset split index to use')
@click.option('--decay-mode', default='qq', type=click.Choice(['qq', 'qqq'], case_sensitive=False), show_default=True,
              help='which decay mode should the signal undergo (qq or qqq)')
@click.option('--dirname',
              default=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                       "LHCO_RnD/tfrecords/samples_{feature_level}"),
              show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('--outdir',
              default="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD_new/CWoLa/RnD_{decay_mode}",
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
    run_cwola(**kwargs)

if __name__ == "__main__":
    cli()