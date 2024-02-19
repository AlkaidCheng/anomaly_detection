#! /usr/bin/env python
from typing import Optional
import os
import glob
import json
import time

import click

from utils import (get_high_level_model, get_low_level_model)
from utils import (get_input_fn, get_filter_fn)
from utils import (get_train_config, get_ws_dataset, get_callbacks)
from utils import (compile_model, load_model, get_model_inputs, freeze_all_layers,
                   get_ws_weight_models)

def run_semi_weakly(mass_point:str, mu:float, decay_mode:str='qq', high_level:bool=True, variables:Optional[str]=None,
                    dataset_index_file:str=("/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/"
                                           "LHCO_RnD_{decay_mode}/tfrecords/sharded_samples/dataset_indices.json"),
                    split_index:int=0,
                    dataset_dir:str="/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/LHCO_RnD_{decay_mode}/tfrecords/sharded_samples",
                    fs_model_dir:str="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD/fully_supervised/RnD_{decay_mode}",
                    outdir:str="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD/weakly_supervised/RnD_{decay_mode}",
                    num_trials:int=10, version_str:str='v1', cache:bool=True, multi_gpu:bool=False):
    import numpy as np
    
    import tensorflow as tf
    
    from quickstats.utils.string_utils import split_str
    from quickstats.utils.common_utils import NpEncoder
    from quickstats.maths.numerics import str_encode_value

    import aliad as ad
    
    m1, m2 = split_str(mass_point, sep=":", remove_empty=True, cast=int)
    dataset_dir = dataset_dir.format(decay_mode=decay_mode)
    fs_model_dir = fs_model_dir.format(decay_mode=decay_mode)
    outdir = outdir.format(decay_mode=decay_mode)
    index_file = dataset_index_file.format(decay_mode=decay_mode)
    var_str = "var_all" if variables is None else "var_" + "_".join(split_str(variables, ",", remove_empty=True))

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

    with open(index_file, 'r') as file:
        ds_index_config = json.load(file)
    ds_indices = ds_index_config[str(split_index)]

    all_ds = get_ws_dataset(m1, m2, mu, high_level=high_level, batch_size=batch_size,
                            dataset_dir=dataset_dir, dataset_indices=ds_indices,
                            variables=variables)

    metadata_filenames = glob.glob(os.path.join(dataset_dir, 'QCD', '*metadata.json'))
    # metadata contains information about the shape and dtype of each type of features, as well as the size of the dataset
    metadata = json.load(open(metadata_filenames[-1]))    

    seed = 1000 * int(1 / mu) + split_index
    np.random.seed(seed)
    random_masses = np.random.uniform(low=0.5, high=6, size=(num_trials, 2)).astype('float32')
    fs_model_path = os.path.join(fs_model_dir,
                                 "high_level" if high_level else "low_level",
                                 "parameterised",
                                 "mass_unordered",
                                 "weighted",
                                 f"SR_10M_events_ratio_1_{var_str}_v1",
                                 f"split_{split_index}",
                                 "full_train.keras")
    if not os.path.exists(fs_model_path):
        raise ValueError(f'Fully supervised model file "{fs_model_path}" does not exist')
        
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            fs_model = load_model(fs_model_path)
            freeze_all_layers(fs_model)
        
            inputs = get_model_inputs(metadata['features'], variables=variables)
            weight_models = get_ws_weight_models(m1=random_masses[0][0],
                                                 m2=random_masses[0][1],
                                                 mu=-1)
            m1_out = 100. * weight_models['m1'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
            m2_out = 100. * weight_models['m2'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
            mu_out = weight_models['mu'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
            mass_params = tf.keras.layers.concatenate([m1_out, m2_out])
            if high_level:
                fs_out = fs_model([inputs['jet_features'], mass_params])
            else:
                fs_out = fs_model([inputs['part_coords'],
                                   inputs['part_features'],
                                   inputs['part_masks'],
                                   inputs['jet_features'], mass_params])
            epsilon = 1e-5
            LLR = fs_out / (1. - fs_out + epsilon)
            LLR_xs = 1.+ mu_out * LLR - mu_out
            ws = LLR_xs / (1. + LLR_xs)
            if high_level:
                ws_model = tf.keras.Model(inputs=inputs['jet_features'], outputs=ws)
            else:
                ws_model = tf.keras.Model(inputs=[inputs['part_coords'],
                                                  inputs['part_features'],
                                                  inputs['part_masks'],
                                                  inputs['jet_features']], outputs=ws)
    else:
        fs_model = load_model(fs_model_path)
        freeze_all_layers(fs_model)
    
        inputs = get_model_inputs(metadata['features'], variables=variables)
        weight_models = get_ws_weight_models(m1=random_masses[0][0],
                                             m2=random_masses[0][1],
                                             mu=-1)
        m1_out = 100. * weight_models['m1'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        m2_out = 100. * weight_models['m2'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        mu_out = weight_models['mu'](tf.ones_like(inputs['jet_features'])[:, 0, 0])
        mass_params = tf.keras.layers.concatenate([m1_out, m2_out])
        if high_level:
            fs_out = fs_model([inputs['jet_features'], mass_params])
        else:
            fs_out = fs_model([inputs['part_coords'],
                               inputs['part_features'],
                               inputs['part_masks'],
                               inputs['jet_features'], mass_params])
        epsilon = 1e-5
        LLR = fs_out / (1. - fs_out + epsilon)
        LLR_xs = 1.+ mu_out * LLR - mu_out
        ws = LLR_xs / (1. + LLR_xs)
        if high_level:
            ws_model = tf.keras.Model(inputs=inputs['jet_features'], outputs=ws)
        else:
            ws_model = tf.keras.Model(inputs=[inputs['part_coords'],
                                              inputs['part_features'],
                                              inputs['part_masks'],
                                              inputs['jet_features']], outputs=ws)
    print(f'##############################################################')
    print(f'INFO: Semi Weakly training with (m1, m2, mu) = ({m1}, {m2}, {mu}) in decay mode {decay_mode}')
    feature_level_txt = 'high level' if high_level else 'low_level'
    print(f'INFO: Training features = {feature_level_txt}')
    if high_level:
        var_txt = 'all' if variables is None else variables
        print(f'INFO: Feature indices = {var_txt}')
    print(f'INFO: Input directory = "{dataset_dir}"')
    print(f'INFO: Output directory = "{outdir}"')
    print(f'##############################################################')

    y_true = None
    total_time = 0.
    
    for i in range(num_trials):
        t0 = time.time()
        trial_checkpoint_dir = os.path.join(checkpoint_dir, f"trial_{i}")
        config = get_train_config(checkpoint_dir=trial_checkpoint_dir, high_level=high_level)
        init_m1, init_m2, init_mu = random_masses[i][0], random_masses[i][1], -3.
        ws_model.trainable_weights[0].assign(tf.fill((1, 1), init_m1))
        ws_model.trainable_weights[1].assign(tf.fill((1, 1), init_m2))
        ws_model.trainable_weights[2].assign(tf.fill((1, 1), init_mu))
        compile_model(ws_model, config)
        callbacks = get_callbacks(config, semi_weakly=True)
        model_filename = os.path.join(config['checkpoint_dir'],
                                      "full_train.keras")
        print(f'--------------------------------------------------------------')
        print(f'(Trial {i}) Initial Weights: m1 = {init_m1}, m2 = {init_m2}, mu = {init_mu}')
        
        # run model training
        if os.path.exists(model_filename) and cache:
            print(f'INFO: Cached model from "{model_filename}"')
            ws_model = tf.keras.models.load_model(model_filename)
        else:
            ws_model.fit(all_ds['train'],
                         validation_data=all_ds['val'],
                         epochs=config['epochs'],
                         callbacks=callbacks)
            print("INFO: Finished training!")
            ws_model.save(model_filename)
        # run prediction
        result_filename = os.path.join(config['checkpoint_dir'], 'test_results.json')
        if os.path.exists(result_filename) and cache:
            print(f'INFO: Cached test results from "{result_filename}"!')
        else:
            predicted_proba = np.concatenate(ws_model.predict(all_ds['test'])).flatten()
            if y_true is None:
                y_true = np.concatenate([y for _, y in all_ds['test']]).flatten()
            print("INFO: Finished prediction!")
            results = {
                'predicted_proba': predicted_proba,
                'y_true': y_true
            }
            with open(result_filename, 'w') as f:
                json.dump(results, f, cls=NpEncoder)
        t1 = time.time()
        dt = t1 - t0
        total_time += dt
        print(f"INFO: Trial finished! Time taken = {dt:.3f} s")
        
    print(f"INFO: Job finished! Total time taken = {total_time:.3f} s")

@click.command(name='run')
@click.option('--mass-point', required=True,
              help='signal mass point to train for (e.g. "500:100")')
@click.option('--mu', required=True, type=float,
              help='signal fraction')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='whether to do training with low-evel or high-level features.')
@click.option('--variables', default=None, show_default=True,
              help='variable indices separated by commas to use in high-level training')
@click.option('--dataset-index-file', default="/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/LHCO_RnD_{decay_mode}/tfrecords/shuffled/dataset_indices.json",
              show_default=True,
              help='config file with dataset split indices')
@click.option('--split-index', default=0, type=int, show_default=True,
              help='which dataset split index to use')
@click.option('--num-trials', default=10, type=int, show_default=True,
              help='number of trials (random mass initialization) to run')
@click.option('--decay-mode', default='qq', type=click.Choice(['qq', 'qqq'], case_sensitive=False), show_default=True,
              help='which decay mode should the signal undergo (qq or qqq)')
@click.option('--dataset-dir',
              default="/pscratch/sd/c/chlcheng/dataset/anomaly_detection/LHC_Olympics_2020/LHCO_RnD_{decay_mode}/tfrecords/sharded_samples",
              show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('--fs-model-dir',
              default="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD/fully_supervised/RnD_{decay_mode}",
              show_default=True,
              help='base directory to which the fully supervised models are located')
@click.option('--outdir',
              default="/pscratch/sd/c/chlcheng/model_checkpoints/LHCO_AD/weakly_supervised/RnD_{decay_mode}",
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
    run_semi_weakly(**kwargs)

if __name__ == "__main__":
    cli()