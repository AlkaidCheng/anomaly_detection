#! /usr/bin/env python
from typing import Optional

import click

from data_loader import DataLoader
from model_loader import ModelLoader
from keywords import FeatureLevel, HIGH_LEVEL, LOW_LEVEL, DEDICATED_SUPERVISED
from keywords import BASE_SEED, DATASET_INDEX_PATH, DATASET_DIR, MODEL_OUTDIRS
from utils import initial_check

def run_dedicated_supervised(mass_point:str,
                             high_level:bool=True,
                             decay_mode:str='qq',
                             variables:Optional[str]=None,
                             dataset_index_file:str=DATASET_INDEX_PATH,
                             split_index:int=0,
                             seed:int=BASE_SEED,
                             batchsize:Optional[int]=None,
                             dataset_dir:str=DATASET_DIR,
                             outdir:str=MODEL_OUTDIRS[DEDICATED_SUPERVISED],
                             cache_dataset:Optional[bool]=None,
                             version_str:str='v1', cache:bool=True,
                             multi_gpu:bool=True,
                             verbosity:str='INFO'):
    import os
    import glob
    import json
    import time
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from quickstats import stdout
    from quickstats.utils.common_utils import NpEncoder
    from quickstats.utils.string_utils import split_str

    t0 = time.time()

    initial_check()

    m1, m2 = split_str(mass_point, sep=':', remove_empty=True, strip=True)
    feature_level = "high_level" if high_level else "low_level"
    var_str = 'var_' + '_'.join(variables.split(",")) if (variables) else 'var_all'
    
    checkpoint_dir = os.path.join(outdir, f"RnD_{decay_mode}", feature_level,
                                  "dedicated", f"W_{decay_mode}_{int(m1)}_{int(m2)}",
                                  f"SR_{var_str}_{version_str}", f"split_{split_index}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_loader = DataLoader(dataset_dir, feature_level, [decay_mode],
                             split_config=dataset_index_file,
                             mass_ordering=False,
                             weighted=False,
                             variables=variables,
                             distributed=False,
                             verbosity=verbosity)
    strategy = data_loader.distribute_strategy
    all_ds = data_loader.get_dedicated_supervised_datasets([m1, m2],
                                                           split_index=split_index,
                                                           seed=seed,
                                                           batchsize=batchsize,
                                                           cache_dataset=cache_dataset)
    feature_metadata = data_loader.feature_metadata
    model_loader = ModelLoader(feature_level,
                               mass_ordering=False,
                               distributed=multi_gpu,
                               strategy=strategy,
                               variables=variables)

    model = model_loader.get_supervised_model(feature_metadata, parametric=False)

    config = model_loader.get_train_config(checkpoint_dir, model_type=DEDICATED_SUPERVISED)

    callbacks = model_loader.get_callbacks(DEDICATED_SUPERVISED, config=config)
    callbacks = [call_back for call_back in callbacks.values()]
    
    model_loader.compile_model(model, config)
    
    model_filename = os.path.join(checkpoint_dir, "full_train.keras")

    stdout.info(f'##############################################################', bare=True)
    stdout.info(f'Dedicated supervised training with (m1, m2, mu) = ({m1}, {m2}) in decay mode {decay_mode}')
    stdout.info(f'Training features = {feature_level}')
    if FeatureLevel.parse(feature_level) == HIGH_LEVEL:
        var_txt = 'all' if variables is None else variables
        stdout.info(f'Feature indices = {var_txt}')
    stdout.info(f'Batchsize = {data_loader._suggest_batchsize(batchsize)}')
    stdout.info(f'Input directory = "{dataset_dir}"')
    stdout.info(f'Output directory = "{outdir}"')
    stdout.info(f'Checkpoint directory = "{checkpoint_dir}"')
    stdout.info(f'##############################################################', bare=True)

    t1 = time.time()

    stdout.info(f'Preparation time = {t1 - t0:.3f}s')

    if data_loader.distributed:
        steps = {dtype: summary['num_batch'] for dtype, summary in data_loader.summary.items()}
    else:
        steps = {dtype: None for dtype in data_loader.summary}
    # run model training
    if os.path.exists(model_filename) and cache:
        stdout.info(f'Cached model from "{model_filename}"')
        model = model_loader.load_model(model_filename)
    else:
        model.fit(all_ds['train'],
                  validation_data=all_ds['val'],
                  epochs=config['epochs'],
                  callbacks=callbacks,
                  steps_per_epoch=steps['train'],
                  validation_steps=steps['val'])
        print("INFO: Finished training!")
        model.save(model_filename)
    t2 = time.time()
    stdout.info(f'Training time = {t2 - t1:.3f}s')
    # release memory
    for ds_type in all_ds:
        if ds_type != 'test':
            all_ds[ds_type] = None
    # run prediction
    result_filename = os.path.join(checkpoint_dir, 'test_results.json')
    if os.path.exists(result_filename) and cache:
        stdout.info(f'Cached test results from "{result_filename}"!')
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
    stdout.info(f'Test time = {t3 - t2:.3f}s')
    stdout.info(f"Job finished! Total time taken = {t3 - t0:.3f}s")  

@click.command(name='run')
@click.option('--mass-point', required=True, show_default=True,
              help='signal mass point to use for training in the form "m1:m2"')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='whether to do training with low-evel or high-level features.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='whether to cache the dataset during training')
@click.option('--variables', default=None, show_default=True,
              help='variable indices separated by commas to use in high-level training')
@click.option('--dataset-index-file', default=DATASET_INDEX_PATH,
              show_default=True,
              help='config file with dataset split indices')
@click.option('--split-index', default=0, type=int, show_default=True,
              help='which dataset split index to use')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='random seed for shuffling dataset')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='batch size')
@click.option('--decay-mode', default='qq', type=click.Choice(['qq', 'qqq'], case_sensitive=False), show_default=True,
              help='which decay mode should the signal undergo (qq or qqq)')
@click.option('--dataset-dir',
              default=DATASET_DIR,
              show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('--outdir',
              default=MODEL_OUTDIRS[DEDICATED_SUPERVISED],
              show_default=True,
              help='base output directory')
@click.option('--version', 'version_str',
              default="v1",
              show_default=True,
              help='version text')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='cache results when applicable')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='whether to use multiple GPUs for training')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='verbosity level ("DEBUG", "INFO", "WARNING", "ERROR")')
def cli(**kwargs):
    run_dedicated_supervised(**kwargs)

if __name__ == "__main__":
    cli()