#! /usr/bin/env python
from typing import Optional

import click

from data_loader import DataLoader
from model_loader import ModelLoader
from utils import initial_check
from keywords import PARAM_SUPERVISED, FeatureLevel, HIGH_LEVEL, LOW_LEVEL
from keywords import DATASET_INDEX_PATH, DATASET_DIR, MODEL_OUTDIRS

def run_param_supervised(high_level:bool=True,
                         decay_mode:str='qq',
                         weighted:bool=True,
                         mass_ordering:bool=False,
                         variables:Optional[str]=None, 
                         include_masses:Optional[str]=None,
                         exclude_masses:Optional[str]=None,
                         interrupt_freq:Optional[int]=None,
                         cycle_bkg_param:bool=True,
                         dataset_index_file:str=DATASET_INDEX_PATH,
                         split_index:int=0,
                         batchsize:Optional[int]=None,
                         dataset_dir:str=DATASET_DIR,
                         outdir:str=MODEL_OUTDIRS[PARAM_SUPERVISED],
                         cache_dataset:Optional[bool]=None,
                         version_str:str='v1', cache:bool=True,
                         multi_gpu:bool=False,
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

    t0 = time.time()
    
    def parse_masses_str(str):
        masses = [m for m in str.split(",") if m]
        masses = [[int(j) for j in k.split(":") if j] for k in masses]
        return masses
    
    if include_masses is not None:
        include_masses = parse_masses_str(include_masses)
        
    if exclude_masses is not None:
        exclude_masses = parse_masses_str(exclude_masses)        

    initial_check()
    
    feature_level = "high_level" if high_level else "low_level"
    var_str = 'var_' + '_'.join(variables.split(",")) if (variables) else 'var_all'
    mass_ordering_str = "mass_ordered" if mass_ordering else "mass_unordered"
    weight_str = "weighted" if weighted else "unweighted"
    checkpoint_dir = os.path.join(outdir, f"RnD_{decay_mode}", feature_level,
                                  "parameterised", mass_ordering_str, weight_str,
                                  f"SR_{var_str}_{version_str}", f"split_{split_index}")
                                  
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_loader = DataLoader(dataset_dir, feature_level, [decay_mode],
                             split_config=dataset_index_file,
                             mass_ordering=mass_ordering,
                             weighted=weighted,
                             variables=variables,
                             distributed=False,
                             verbosity=verbosity)
    all_ds = data_loader.get_param_supervised_datasets(split_index=split_index,
                                                       batchsize=batchsize,
                                                       include_masses=include_masses,
                                                       exclude_masses=exclude_masses,
                                                       cache_dataset=cache_dataset)
    strategy = data_loader.distribute_strategy
    feature_metadata = data_loader.feature_metadata
    model_loader = ModelLoader(feature_level,
                               mass_ordering=mass_ordering,
                               distributed=multi_gpu,
                               strategy=strategy,
                               variables=variables)

    model = model_loader.get_supervised_model(feature_metadata, parametric=True)

    config = model_loader.get_train_config(checkpoint_dir, model_type=PARAM_SUPERVISED)
    config['callbacks']['early_stopping']['interrupt_freq'] = interrupt_freq
    
    callbacks = model_loader.get_callbacks(PARAM_SUPERVISED, config=config)
    early_stopping = callbacks['early_stopping']
    callbacks = [callback for callback in callbacks.values()]
    
    model_loader.compile_model(model, config)
       
    if interrupt_freq:
        model_loader.restore_model(early_stopping, model, checkpoint_dir)

    model_filename = model_loader.get_model_save_path(checkpoint_dir)
    
    stdout.info(f'##############################################################', bare=True)
    stdout.info(f'Param supervised training with in decay mode {decay_mode}')
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
                  initial_epoch=early_stopping.initial_epoch,
                  steps_per_epoch=steps['train'],
                  validation_steps=steps['val'])
        if early_stopping.interrupted:
            stdout.info(f'Training interrupted!')
            return None
        stdout.info(f'Finished training!')
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
        if weighted:
            y_true = np.concatenate([y for _, y, _ in all_ds['test']]).flatten()
        else:
            y_true = np.concatenate([y for _, y in all_ds['test']]).flatten()
                
        stdout.info(f'Finished prediction!')
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
    stdout.info(f'Test time = {t3 - t2:.3f}s')
    stdout.info(f'Job finished! Total time taken = {t3 - t0:.3f}s')

    

@click.command(name='run')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='whether to do training with low-evel or high-level features.')
@click.option('--weighted/--unweighted', default=True, show_default=True,
              help='whether to use sample weights in training')
@click.option('--mass-ordering/--no-mass-ordering', default=False, show_default=True,
              help='whether to order the parametric masses')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='whether to cache the dataset during training')
@click.option('--variables', default=None, show_default=True,
              help='variable indices separated by commas to use in high-level training')
@click.option('--dataset-index-file', default=DATASET_INDEX_PATH,
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
@click.option('--dataset-dir',
              default=DATASET_DIR,
              show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('--outdir',
              default=MODEL_OUTDIRS[PARAM_SUPERVISED],
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
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='verbosity level ("DEBUG", "INFO", "WARNING", "ERROR")')              
def cli(**kwargs):
    run_param_supervised(**kwargs)

if __name__ == "__main__":
    cli()