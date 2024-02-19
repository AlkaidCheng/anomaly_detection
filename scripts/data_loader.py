from typing import Optional, Dict, List, Union
import os
import glob
import json

import numpy as np
import pandas as pd

from keywords import FeatureLevel, HIGH_LEVEL, LOW_LEVEL
from keywords import DecayMode, TWO_PRONG, THREE_PRONG
from keywords import ModelType, DEDICATED_SUPERVISED, PARAM_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY
from keywords import MassOrdering, MASS_UNORDERED, MASS_ORDERED
from keywords import (SIG_SAMPLE, BKG_SAMPLE, EXT_BKG_SAMPLE,
                      PARAM_SIG_SAMPLE, PARAM_BKG_SAMPLE, PARAM_EXT_BKG_SAMPLE)
from keywords import TRAIN_FEATURES, WEIGHT_FEATURES, PARAM_FEATURES
from keywords import NUM_SHARDS, BASE_SEED

from quickstats import AbstractObject

class DataLoader(AbstractObject):

    ZERO_BKG_LABEL = 'QCD (zero-labeled)'
    ONE_BKG_LABEL = 'QCD (one-labeled)'
    SIG_LABEL = 'W ({decay_mode})'

    @property
    def feature_level(self):
        return self._feature_level

    @feature_level.setter
    def feature_level(self, value:str):
        feature_level = FeatureLevel.parse(value)
        self._feature_level = feature_level

    @property
    def decay_modes(self):
        return self._decay_modes

    @decay_modes.setter
    def decay_modes(self, values:List[str]):
        decay_modes = [DecayMode.parse(v) for v in values]
        assert len(decay_modes) in [1, 2]
        self._decay_modes = decay_modes

    @property
    def mass_ordering(self):
        return self._mass_ordering

    @mass_ordering.setter
    def mass_ordering(self, value:str):
        mass_ordering = MassOrdering.parse(value)
        self._mass_ordering = mass_ordering

    def __init__(self, data_dir:str,
                 feature_level:str,
                 decay_modes:List[str],
                 split_config:Optional[Union[str, Dict]]=None,
                 mass_ordering:bool=MASS_UNORDERED,
                 weighted:bool=False,
                 verbosity:str='INFO'):
        super().__init__(verbosity=verbosity)
        self.data_dir = data_dir
        self.feature_level = feature_level
        self.decay_modes = decay_modes
        self.mass_ordering = mass_ordering
        self.weighted = weighted
        self.data_dir = data_dir
        self.set_split_config(split_config)

    def generate_split_config(self):
        np.random.seed(BASE_SEED)
        config = {}
        train_size = NUM_SHARDS//2
        test_size = NUM_SHARDS//4
        for i in range(100):
            if i == 0:
                indices = np.arange(NUM_SHARDS)
            else:
                indices = np.random.permutation(NUM_SHARDS)
            config[i] = {
                'train' : indices[:train_size],
                'val'   : indices[train_size: train_size + test_size],
                'test'  : indices[train_size + test_size:]
            }
        return config        

    def set_split_config(self, config:Optional[Union[str, Dict]]=None):
        if config is None:
            config = self.generate_split_config()
        elif isinstance(config, str):
            with open(config, 'r') as file:
                config = json.load(file)
                config = {int(k):v for k, v in config.items()}
        else:
            config = dict(config)
        self.split_config = config

    def _get_weight_feature(self):
        return WEIGHT_FEATURES[self.mass_ordering]

    def _get_param_feature(self):
        return PARAM_FEATURES[self.mass_ordering]

    def _suggest_batchsize(self, batchsize:Optional[int]=None):
        if batchsize is None:
            batchsize = 1024 if (self.feature_level == HIGH_LEVEL) else 128
        return batchsize

    def _suggest_cache_dataset(self, cache_dataset:Optional[bool]=None):
        if cache_dataset is None:
            cache_dataset = True if (self.feature_level == HIGH_LEVEL) else False
        return cache_dataset

    def _get_train_features(self, model_type:ModelType):
        model_type = ModelType.parse(model_type)
        features = list(TRAIN_FEATURES[self.feature_level])
        if model_type == PARAM_SUPERVISED:
            param_feature = self._get_param_feature()
            features.append(param_feature)
        return features

    def _get_aux_features(self, model_type:ModelType):
        model_type = ModelType.parse(model_type)
        features = ['label']
        if model_type in [IDEAL_WEAKLY, SEMI_WEAKLY]:
            return features
        if self.weighted:
            weight_feature = self._get_weight_feature()
            features.append(weight_feature)
        return features
        
    def _get_required_features(self, model_type:ModelType):
        model_type = ModelType.parse(model_type)
        train_features = self._get_train_features(model_type=model_type)
        aux_features = self._get_aux_features(model_type=model_type)
        features = train_features + aux_features
        return features

    def _get_filter_fn(self, masses, mode:str="include", ignore_bkg:bool=False):
        import tensorflow as tf
        masses_tensor = tf.constant(masses, dtype='float64')
        param_feature = self._get_param_feature()
        if mode == "exclude":
            if ignore_bkg:
                def filter_fn(x):
                    return ((x['label'][0] == 0) or 
                    tf.reduce_all(tf.reduce_any(tf.not_equal(x[param_feature], masses_tensor), axis=1)))
            else:
                def filter_fn(x):
                    return tf.reduce_all(tf.reduce_any(tf.not_equal(x[param_feature], masses_tensor), axis=1))
        elif mode == "include":
            if ignore_bkg:
                def filter_fn(x):
                    return ((x['label'][0] == 0) or 
                    tf.reduce_any(tf.reduce_all(tf.equal(x[param_feature], masses_tensor), axis=1)))
            else:
                def filter_fn(x):
                    return tf.reduce_any(tf.reduce_all(tf.equal(x[param_feature], masses_tensor), axis=1))
        else:
            raise ValueError(f"unknwon filter mode: {mode}")
        return filter_fn

    def _decorate_dataset_specs(self, specs):
        df = pd.DataFrame(specs)
        get_metadata_path = lambda f: f'{os.path.splitext(f)[0]}_metadata.json'
        get_shard_index = lambda f: int(os.path.splitext(f)[0].split("_")[-1])
        get_dataset_size = lambda f: json.load(open(f))['size']
        df['metadata_path'] = df['dataset_path'].apply(get_metadata_path)
        df['shard_index'] = df['dataset_path'].apply(get_shard_index)
        if 'sample' in df.columns:
            df = df.sort_values(['sample', 'shard_index'])
        else:
            df = df.sort_values('shard_index')
        df = df.reset_index(drop=True)
        df['size'] = df['metadata_path'].apply(get_dataset_size)
        return df
        
    def get_dedicated_dataset_specs(self, mass_point, samples:Optional[List[str]]=None):
        m1, m2 = mass_point
        if samples is None:
            samples = [PARAM_SIG_SAMPLE, PARAM_BKG_SAMPLE, PARAM_EXT_BKG_SAMPLE]
        resolved_samples = []
        for sample in samples:
            for decay_mode in self.decay_modes:
                resolved_sample = sample.format(m1=m1, m2=m2, decay_mode=decay_mode.key)
                if resolved_sample not in resolved_samples:
                    resolved_samples.append(resolved_sample)
        specs = {
            "dataset_path" : [],
            "sample": []
        }
        dirname = os.path.join(self.data_dir, f'samples_{self.feature_level.key}')
        for sample in resolved_samples:
            sample_dir = os.path.join(dirname, sample)
            if not os.path.exists(sample_dir):
                raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")
            dataset_paths = glob.glob(os.path.join(sample_dir, "*.tfrec"))
            if not dataset_paths:
                raise RuntimeError(f"no dataset files found for the sample '{sample}' "
                                   f"under the directory '{sample_dir}'")
            specs['dataset_path'].extend(dataset_paths)
            specs['sample'].extend([sample]*len(dataset_paths))
        df = self._decorate_dataset_specs(specs)
        return df

    def get_param_dataset_specs(self):
        specs = {
            "dataset_path" : [],
            "sample"       : []
        }
        dirname = os.path.join(self.data_dir, f'shuffled_{self.feature_level.key}')
        dataset_filenames = glob.glob(os.path.join(dirname, "*.tfrec"))
        if not dataset_filenames:
            raise RuntimeError(f"no dataset files found under the directory '{dirname}'")
        specs['dataset_path'].extend(dataset_filenames)
        specs['sample'].extend(["mixed"] * len(dataset_filenames))
        df = self._decorate_dataset_specs(specs)
        return df

    def _get_all_filters(self,
                         include_masses:Optional[List[List[float]]]=None,
                         exclude_masses:Optional[List[List[float]]]=None):
        filters = []
        if include_masses:
            filter_incl = self._get_filter_fn(include_masses, mode='include', ignore_bkg=False)
            filters.append(filter_incl)
        if exclude_masses:
            filter_excl = self._get_filter_fn(exclude_masses, mode='exclude', ignore_bkg=False)
            filters.append(filter_excl)
        return filters

    def _get_all_transforms(self, model_type:ModelType,
                            custom_masses:Optional[List[float]]=None,
                            custom_label:Optional[int]=None,
                            variables:Optional[str]=None):
        import tensorflow as tf
        from aliad.interface.tensorflow.dataset import feature_selector
        
        transforms = []
        
        if custom_label is not None:
            if custom_label == 0:
                value = tf.zeros((1,), dtype='int64')
            elif custom_label == 1:
                value = tf.ones((1,), dtype='int64')
            else:
                raise ValueError('custom_label must be either 0 or 1')
            def modify_label(X, value=value):
                X['label'] = value
                return X
            transforms.append(modify_label)
            
        if custom_masses is not None:
            m1, m2 = custom_masses
            param_feature = self._get_param_feature()
            value = tf.constant([m1, m2], dtype='float64')
            def modify_param_masses(X, value=value):
                X[param_feature] = value
                return X
            transforms.append(modify_param_masses)
            
        if variables is not None:
            var_index = tf.constant([int(i) for i in variables.split(",") if i])
            def select_jet_feature(X):
                X['jet_features'] = tf.gather(X['jet_features'], var_index, axis=1)
                return X
            transforms.append(select_jet_feature)
            
        model_type = ModelType.parse(model_type)
        train_features = self._get_train_features(model_type)
        # sometimes we also want the mass parameters to be included in dedicated dataset
        if custom_masses is not None:
            param_feature = self._get_param_feature()
            if param_feature not in train_features:
                train_features.append(param_feature)
        aux_features = self._get_aux_features(model_type)
        input_fn = feature_selector(train_features, aux_features)
        transforms.append(input_fn)
        
        return transforms

    def _print_size_summary(self, size_summary:Dict):
        df = pd.DataFrame(size_summary)
        if 'mixed' in df.index:
            df = df.drop(index=['mixed'])
        self.stdout.info('Number of events in each dataset splits:')
        self.stdout.info(df, bare=True)

    def get_supervised_datasets(self, mass_point:Optional[List[float]]=None,
                                split_index:int=0,
                                samples:Optional[List[str]]=None,
                                variables:Optional[str]=None,
                                batchsize:Optional[int]=None,
                                custom_masses:Optional[List[float]]=None,
                                include_masses:Optional[str]=None,
                                exclude_masses:Optional[str]=None,
                                cache_dataset:Optional[bool]=None):
        from aliad.interface.tensorflow.dataset import get_ndarray_tfrecord_example_parser, apply_pipelines
        split_config = self.split_config[split_index]
        model_type = PARAM_SUPERVISED if (mass_point is None) else DEDICATED_SUPERVISED
        if model_type == PARAM_SUPERVISED:
            df = self.get_param_dataset_specs()
        else:
            df = self.get_dedicated_dataset_specs(mass_point, samples=samples)
        with open(df['metadata_path'].iloc[0], 'r') as file:
            metadata = json.load(file)
        required_features = self._get_required_features(model_type)
        parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'],
                                                                keys=required_features)
        filters = self._get_all_filters(include_masses=include_masses,
                                        exclude_masses=exclude_masses)
        transforms = self._get_all_transforms(model_type, variables=variables,
                                              custom_masses=custom_masses)
        batchsize = self._suggest_batchsize(batchsize)
        cache_dataset = self._suggest_cache_dataset(cache_dataset)

        import tensorflow as tf
        stages = list(split_config)
        all_ds = {}
        size_summary = {}
        samples = df['sample'].unique()
        for stage in stages:
            size_summary[stage] = {}
            stage_mask = df['shard_index'].isin(split_config[stage])
            stage_df = df[stage_mask]
            total_size = 0
            for sample in samples:
                sample_size = stage_df[stage_df['sample'] == sample]['size'].sum()
                total_size += sample_size
                size_summary[stage][sample] = sample_size
            size_summary[stage]['total'] = total_size
            dataset_paths = stage_df['dataset_path']
            ds = tf.data.TFRecordDataset(dataset_paths, num_parallel_reads=tf.data.AUTOTUNE)
            ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)     
            for transform in transforms:
                ds = ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
            shuffle = (model_type == DEDICATED_SUPERVISED) and (stage == 'train')
            buffer_size = total_size if shuffle else None
            cache = cache_dataset and (stage != 'test')
            ds = apply_pipelines(ds, batch_size=batchsize,
                                 cache=cache,
                                 shuffle=shuffle,
                                 prefetch=True,
                                 buffer_size=buffer_size,
                                 drop_remainder=False,
                                 reshuffle_each_iteration=False)
            all_ds[stage] = ds
        self._print_size_summary(size_summary)
        return all_ds
        
    def get_dedicated_supervised_datasets(self, mass_point:List[float],
                                          split_index:int=0,
                                          samples:Optional[List[str]]=None,
                                          variables:Optional[str]=None,
                                          custom_masses:Optional[List[float]]=None,
                                          batchsize:Optional[int]=None,
                                          cache_dataset:Optional[bool]=None):
        if mass_point is None:
            raise ValueError('mass_point must be specified for dedicated training')
        all_ds = self.get_supervised_datasets(mass_point=mass_point,
                                              split_index=split_index,
                                              samples=samples,
                                              variables=variables,
                                              custom_masses=custom_masses,
                                              batchsize=batchsize,
                                              cache_dataset=cache_dataset)
        return all_ds

    def get_param_supervised_datasets(self, split_index:int=0,
                                      variables:Optional[str]=None,
                                      batchsize:Optional[int]=None,
                                      include_masses:Optional[str]=None,
                                      exclude_masses:Optional[str]=None,
                                      cache_dataset:Optional[bool]=None):
        all_ds = self.get_supervised_datasets(mass_point=None,
                                              split_index=split_index,
                                              variables=variables,
                                              batchsize=batchsize,
                                              include_masses=include_masses,
                                              exclude_masses=exclude_masses,
                                              cache_dataset=cache_dataset)
        return all_ds

    def _get_signal_injection_composition(self, df:"pandas.DataFrame", mu:float, alpha:float=0):
        if (mu is not None) and (not ((mu > 0) and (mu < 1))):
            raise ValueError("signal fraction (mu) must be in the range (0, 1)")
        if not ((alpha >= 0) and (alpha < 1)):
            raise ValueError("branching fraction (alpha) must be in the range [0, 1)")
        df = df.copy()
        composition = []
        sample_str = df['sample'].str
        sample_masks = {
            'bkg': (sample_str.contains(BKG_SAMPLE)) | (sample_str.contains(EXT_BKG_SAMPLE)),
        }
        sample_sizes = {'sig': 0}
        df['sample_type'] = 'unknown'
        for decay_mode in self.decay_modes:
            sample_masks[f'sig_{decay_mode.key}'] = sample_str.contains(SIG_SAMPLE.format(decay_mode=decay_mode.key))
        for key, masks in sample_masks.items():
            df.loc[masks, ['sample_type']] = key
            sample_sizes[key] = df[masks]['size'].sum()
            if key.startswith('sig_'):
                sample_sizes['sig'] += sample_sizes[key]
        df = df.sort_values(['sample_type', 'shard_index', 'sample'])
        df = df.reset_index(drop=True)
        total_bkg_size = sample_sizes['bkg']
        total_sig_size = sample_sizes['sig']
        if not total_bkg_size:
            raise RuntimeError('no background sample(s) included in the dataset')
        if not total_sig_size:
            raise RuntimeError('no signal sample(s) included in the dataset')
        if mu is not None:
            exp_sig_size = int(total_bkg_size * mu)
            if exp_sig_size > total_sig_size:
                exp_bkg_size = int(total_sig_size / mu)
                self.stdout.warning(f"Number of available signal events ({total_sig_size}) not enough to compose "
                                    f"dataset with a signal fraction of {mu} (requires {exp_sig_size} events). Will shrink "
                                    f"size of background events from {total_bkg_size} to {exp_bkg_size}", "red")
                exp_sig_size = total_sig_size
            else:
                exp_bkg_size = total_bkg_size
            num_modes = len(self.decay_modes)
            exp_decay_mode_sig_sizes = {}
            total_decay_mode_sig_sizes = {}
            if num_modes == 2:
                decay_modes = [decay_mode for decay_mode in DecayMode]
                branching_ratios = [(1 - alpha), alpha]
                for i in range(len(decay_modes)):
                    decay_mode = decay_modes[i]
                    exp_size = int(exp_sig_size * branching_ratios[i])
                    total_size =  df[df['sample_type'] == f'sig_{decay_mode.key}']['size'].sum()
                    if exp_size > total_size:
                        exp_sig_size = np.floor(size / branching_ratios[i])
                        exp_bkg_size = int(exp_sig_size / mu)
                        self.stdout.warning(f"Number of available {decay_mode.key} signal events "
                                            f"({total_size}) not enough to compose dataset with a "
                                            f"signal fraction of {mu} with branching fraction "
                                            f"{branching_ratios[i]}(requires {exp_size} events). "
                                            f"Will shrink size of signal events from "
                                            f" {total_sig_size} to {exp_sig_size} and size of "
                                            f"background events from {total_bkg_size} to {exp_bkg_size}", "red")
                        exp_size = int(exp_sig_size * branching_ratios[i])
                    exp_decay_mode_sig_sizes[decay_mode].append(exp_size)
                    total_decay_mode_sig_sizes[decay_mode].append(total_size)
            elif num_modes == 1:
                exp_decay_mode_sig_sizes[self.decay_modes[0]] = exp_sig_size
                total_decay_mode_sig_sizes[self.decay_modes[0]] = total_sig_size
            else:
                raise RuntimeError(f'unexpected number of signal decay modes: {num_modes}')
            ref_size = int(exp_bkg_size / 2)
            data_size = exp_bkg_size - ref_size
        # take all signals
        else:
            ref_size = total_bkg_size
            data_size = 0
            exp_decay_mode_sig_sizes = {}
            for decay_mode in self.decay_modes:
                exp_decay_mode_sig_sizes[decay_mode] = sample_sizes[f'sig_{decay_mode.key}']
        bkg_dataset_paths = df[df['sample_type'] == 'bkg']['dataset_path'].values
        composition.append({
            'dataset_paths': bkg_dataset_paths,
            'components': [
                {
                    'name': self.ZERO_BKG_LABEL,
                    'label': 0,
                    'skip': 0,
                    'take': ref_size
                },
                {
                    'name': self.ONE_BKG_LABEL,
                    'label': 1,
                    'skip': ref_size,
                    'take': data_size
                }
            ]
        })
        for decay_mode, exp_size in exp_decay_mode_sig_sizes.items():
            sig_dataset_paths = df[df['sample_type'] == f'sig_{decay_mode.key}']['dataset_path'].values
            composition.append({
                'dataset_paths': sig_dataset_paths,
                'components': [
                    {
                        'name': self.SIG_LABEL.format(decay_mode=decay_mode.key),
                        'label': 1,
                        'skip': 0,
                        'take': exp_size
                    }
                ]
            })
        return composition

    def get_weakly_datasets(self, mass_point:List[float],
                            mu:float, alpha:float=0,
                            split_index:int=0,
                            samples:Optional[List[str]]=None,
                            variables:Optional[str]=None,
                            batchsize:Optional[int]=None,
                            cache_dataset:Optional[bool]=None):
        from aliad.interface.tensorflow.dataset import (get_ndarray_tfrecord_example_parser,
                                                        concatenate_datasets,
                                                        apply_pipelines)
        split_config = self.split_config[split_index]
        df = self.get_dedicated_dataset_specs(mass_point, samples=samples)
        with open(df['metadata_path'].iloc[0], 'r') as file:
            metadata = json.load(file)
        required_features = self._get_required_features(SEMI_WEAKLY)
        parse_tfrecord_fn = get_ndarray_tfrecord_example_parser(metadata['features'],
                                                                keys=required_features)
        batchsize = self._suggest_batchsize(batchsize)
        cache_dataset = self._suggest_cache_dataset(cache_dataset)        

        import tensorflow as tf
        stages = list(split_config)
        all_ds = {}
        size_summary = {}
        samples = df['sample'].unique()
        do_mixed_signals = len(self.decay_modes) == 2
        transforms = {
            0: self._get_all_transforms(SEMI_WEAKLY, variables=variables,
                                        custom_label=0),
            1: self._get_all_transforms(SEMI_WEAKLY, variables=variables,
                                        custom_label=1)
        }
        for stage in stages:
            size_summary[stage] = {}
            total_size = 0
            stage_mask = df['shard_index'].isin(split_config[stage])
            stage_df = df[stage_mask].copy()
            stage_mu = mu if stage != 'test' else None
            sample_composition = self._get_signal_injection_composition(stage_df, mu=stage_mu, alpha=alpha)
            datasets = []
            for composition in sample_composition:
                ds = tf.data.TFRecordDataset(composition['dataset_paths'],
                                             num_parallel_reads=tf.data.AUTOTUNE)
                ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
                for component in composition['components']:
                    size_summary[stage][component['name']] = component['take']
                    total_size += component['take']
                    if (component['take'] == 0):
                        continue
                    ds_i = ds
                    if (component['skip'] > 0):
                        ds_i = ds_i.skip(component['skip'])
                    ds_i = ds_i.take(component['take'])
                    for transform in transforms[component['label']]:
                        ds_i = ds_i.map(transform)
                    datasets.append(ds_i)
            size_summary[stage]['total'] = total_size
            ds = concatenate_datasets(datasets)
            shuffle = (stage == 'train')
            buffer_size = total_size if shuffle else None
            cache = cache_dataset and (stage != 'test')
            ds = apply_pipelines(ds, batch_size=batchsize,
                                 cache=cache,
                                 shuffle=shuffle,
                                 prefetch=True,
                                 buffer_size=buffer_size,
                                 drop_remainder=False,
                                 reshuffle_each_iteration=False)
            all_ds[stage] = ds
        self._print_size_summary(size_summary)
        return all_ds





