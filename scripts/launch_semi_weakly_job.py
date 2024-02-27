#! /usr/bin/env python
from typing import Optional

import click

from data_loader import DataLoader
from model_loader import ModelLoader
from utils import initial_check
from keywords import DecayMode, FeatureLevel, HIGH_LEVEL, LOW_LEVEL, SEMI_WEAKLY, PARAM_SUPERVISED
from keywords import DATASET_INDEX_PATH, DATASET_DIR, MODEL_OUTDIRS

def run_semi_weakly(mass_point:str, mu:float,
                    alpha:Optional[float]=None,
                    high_level:bool=True,
                    decay_mode:str='qq',
                    decay_mode_2:Optional[str]=None,
                    variables:Optional[str]=None,
                    dataset_index_file:str=DATASET_INDEX_PATH,
                    split_index:int=0,
                    batchsize:Optional[int]=None,
                    dataset_dir:str=DATASET_DIR,
                    fs_model_dir:str=MODEL_OUTDIRS[PARAM_SUPERVISED],
                    outdir:str=MODEL_OUTDIRS[SEMI_WEAKLY],
                    num_trials:int=5,
                    version_str:str='v1',
                    cache_dataset:Optional[bool]=None,
                    cache:bool=True,
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
    from quickstats.utils.string_utils import split_str
    from quickstats.utils.common_utils import NpEncoder
    from quickstats.maths.numerics import str_encode_value

    t0 = time.time()    

    initial_check()

    m1, m2 = split_str(mass_point, sep=":", remove_empty=True, cast=int)
    decay_modes = [decay_mode]
    if decay_mode_2 is not None:
        decay_modes.append(decay_mode_2)
    # ensure correct ordering
    if len(decay_modes) == 2:
        decay_modes = [decay_mode.key for decay_mode in DecayMode]
        mixed_signal = True
    else:
        mixed_signal = False

    feature_level = "high_level" if high_level else "low_level"
    var_str = 'var_' + '_'.join(variables.split(",")) if (variables) else 'var_all'
    decay_mode_str = "_".join(decay_modes)
    mu_alpha_str = f"mu_{str_encode_value(mu)}"
    if mixed_signal:
        mu_alpha_str += f"_alpha_{str_encode_value(alpha)}"
    checkpoint_dir = os.path.join(outdir, f"RnD_{decay_mode}", feature_level,
                                  f"W_{decay_mode_str}_{int(m1)}_{int(m2)}",
                                  f"SR_{var_str}_{version_str}",
                                  mu_alpha_str,
                                  f"split_{split_index}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_loader = DataLoader(dataset_dir, feature_level, decay_modes,
                             split_config=dataset_index_file,
                             mass_ordering=False,
                             weighted=False,
                             variables=variables,
                             distributed=False,
                             verbosity=verbosity)
    strategy = data_loader.distribute_strategy
    all_ds = data_loader.get_weakly_datasets([m1, m2],
                                             mu=mu,
                                             alpha=alpha,
                                             split_index=split_index,
                                             batchsize=batchsize,
                                             cache_dataset=cache_dataset)

    feature_metadata = data_loader.feature_metadata
    model_loader = ModelLoader(feature_level,
                               mass_ordering=False,
                               distributed=multi_gpu,
                               strategy=strategy,
                               variables=variables)

    seed = 1000 * int(1 / mu) + split_index
    np.random.seed(seed)
    random_masses = np.random.uniform(low=0.5, high=6, size=(num_trials, 2)).astype('float32')
    fs_model_path = os.path.join(fs_model_dir,
                                 f"RnD_{decay_modes[0]}",
                                 feature_level,
                                 "parameterised",
                                 "mass_unordered",
                                 "unweighted",
                                 f"SR_{var_str}_{version_str}",
                                 f"split_{split_index}",
                                 "full_train.keras")
    if not os.path.exists(fs_model_path):
        raise ValueError(f'Fully supervised model file "{fs_model_path}" does not exist')
    if len(decay_modes) == 2:
        fs_model_path_2 = os.path.join(fs_model_dir,
                                     f"RnD_{decay_modes[1]}",
                                     feature_level,
                                     "parameterised",
                                     "mass_unordered",
                                     "unweighted",
                                     f"SR_{var_str}_{version_str}",
                                     f"split_{split_index}",
                                     "full_train.keras")
        if not os.path.exists(fs_model_path_2):
            raise ValueError(f'Fully supervised model file "{fs_model_path_2}" does not exist')
    else:
        fs_model_path_2 = None

    ws_model = model_loader.get_semi_weakly_model(feature_metadata,
                                                  m1=random_masses[0][0],
                                                  m2=random_masses[0][1],
                                                  fs_model_path=fs_model_path,
                                                  fs_model_path_2=fs_model_path_2)

    stdout.info(f'##############################################################', bare=True)
    stdout.info(f'Semi Weakly training with (m1, m2, mu) = ({m1}, {m2}, {mu}) in decay mode(s) {decay_modes}')
    stdout.info(f'Training features = {feature_level}')
    if FeatureLevel.parse(feature_level) == HIGH_LEVEL:
        var_txt = 'all' if variables is None else variables
        stdout.info(f'Feature indices = {var_txt}')
    stdout.info(f'Batchsize = {data_loader._suggest_batchsize(batchsize)}')
    stdout.info(f'Input directory = "{dataset_dir}"')
    stdout.info(f'Output directory = "{outdir}"')
    stdout.info(f'Checkpoint directory = "{checkpoint_dir}"')
    stdout.info(f'Supervised model path ({decay_modes[0]}) = "{fs_model_path}"')
    if fs_model_path_2 is not None:
        stdout.info(f'Supervised model path ({decay_modes[1]}) = "{fs_model_path_2}"')
    stdout.info(f'##############################################################', bare=True)

    t1 = time.time()
    stdout.info(f'Preparation time = {t1 - t0:.3f}s')

    if data_loader.distributed:
        steps = {dtype: summary['num_batch'] for dtype, summary in data_loader.summary.items()}
    else:
        steps = {dtype: None for dtype in data_loader.summary}

    y_true = None
    total_time = 0.
    
    for i in range(num_trials):
        ti_0 = time.time()
        trial_checkpoint_dir = os.path.join(checkpoint_dir, f"trial_{i}")
        config = model_loader.get_train_config(trial_checkpoint_dir, model_type=SEMI_WEAKLY)
        callbacks = model_loader.get_callbacks(SEMI_WEAKLY, config=config)
        callbacks = [callback for callback in callbacks.values()]
        model_loader.compile_model(ws_model, config)
        model_filename = model_loader.get_model_save_path(trial_checkpoint_dir)
        init_m1, init_m2, init_mu = random_masses[i][0], random_masses[i][1], -3
        init_alpha = 0.5 if mixed_signal else None
        model_loader.set_semi_weakly_model_weights(ws_model, m1=init_m1, m2=init_m2,
                                                   mu=init_mu, alpha=init_alpha)
        weight_str = f"m1 = {init_m1}, m2 = {init_m2}, mu = {init_mu}"
        if init_alpha is not None:
            weight_str += f", alpha = {init_alpha}"
        stdout.info(f'--------------------------------------------------------------', bare=True)
        stdout.info(f'(Trial {i}) Initial Weights: {weight_str}')
        # run model training
        if os.path.exists(model_filename) and cache:
            stdout.info(f'Cached model from "{model_filename}"')
            ws_model = model_loader.load_model(model_filename)
        else:
            ws_model.fit(all_ds['train'],
                         validation_data=all_ds['val'],
                         epochs=config['epochs'],
                         callbacks=callbacks,
                         steps_per_epoch=steps['train'],
                         validation_steps=steps['val'])
            stdout.info('Finished training!')
            ws_model.save(model_filename)
        # run prediction
        result_filename = os.path.join(config['checkpoint_dir'], 'test_results.json')
        if os.path.exists(result_filename) and cache:
            stdout.info(f'Cached test results from "{result_filename}"')
        else:
            predicted_proba = np.concatenate(ws_model.predict(all_ds['test'])).flatten()
            if y_true is None:
                y_true = np.concatenate([y for _, y in all_ds['test']]).flatten()
            stdout.info('Finished prediction!')
            results = {
                'predicted_proba': predicted_proba,
                'y_true': y_true
            }
            with open(result_filename, 'w') as f:
                json.dump(results, f, cls=NpEncoder)
        ti_1 = time.time()
        dt = ti_1 - ti_0
        total_time += dt
        stdout.info(f'Trial finished! Time taken = {dt:.3f} s')
    total_time += (t1 - t0)
    stdout.info(f'Job finished! Total time taken = {total_time:.3f} s')

@click.command(name='run')
@click.option('--mass-point', required=True,
              help='signal mass point to train for (e.g. "500:100")')
@click.option('--mu', required=True, type=float,
              help='signal fraction')
@click.option('--alpha', default=0., type=float,
              help='signal branching fraction')              
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
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='batch size')              
@click.option('--num-trials', default=5, type=int, show_default=True,
              help='number of trials (random mass initialization) to run')
@click.option('--decay-mode', default='qq', type=click.Choice(['qq', 'qqq'],
              case_sensitive=False), show_default=True,
              help='which decay mode should the signal undergo (qq or qqq)')
@click.option('--decay-mode-2', default=None, type=click.Choice(['qq', 'qqq'],
              case_sensitive=False), show_default=True,
              help='additional decay mode of the signal (qq or qqq)')      
@click.option('--dataset-dir',
              default=DATASET_DIR,
              show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('--fs-model-dir',
              default=MODEL_OUTDIRS[PARAM_SUPERVISED],
              show_default=True,
              help='base directory to which the fully supervised models are located')          
@click.option('--outdir',
              default=MODEL_OUTDIRS[SEMI_WEAKLY],
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
    run_semi_weakly(**kwargs)

if __name__ == "__main__":
    cli()