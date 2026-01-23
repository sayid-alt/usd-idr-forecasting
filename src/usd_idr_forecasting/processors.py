import os
import shutil
import pickle
import wandb
import numpy as np
import pandas as pd
import tensorflow as tf

from loguru import logger
from typing import Union
from dotenv import load_dotenv
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.utils import wandb_auth, get_dt_now
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

load_dotenv()
PROJECT_WORKING_DIR = os.getenv('PROJECT_WORKING_DIR')
WANDB_IDRX_FORECAST_KEY = os.getenv('WANDB_IDRX_FORECAST_KEY')
wandb_auth(key=WANDB_IDRX_FORECAST_KEY)


class Scaler(BaseEstimator, TransformerMixin):
	"""Scaler class to normalize time series data using MinMaxScaler"""
	def __init__(self, scaler_obj=None):
		self._scaler = scaler_obj if scaler_obj is not None else MinMaxScaler()

	def fit(self, X: pd.Series, y=None):
		self._scaler.fit(X.values.reshape(-1, 1))
		return self

	def transform(self, X: pd.Series) -> pd.Series:
		X_scaled = self._scaler.transform(X.values.reshape(-1, 1))
		return pd.Series(X_scaled.flatten(), index=X.index)
	
	def inverse_transform(self, X):
		return self._scaler.inverse_transform(X)

class Windower(BaseEstimator, TransformerMixin):
	"""Windower class to create windowed dataset for time series forecasting"""
	def __init__(
		self,
		window_size: int, 
		batch_size: int,
		target_size: int,
		shuffle_buffer: int = None,
	):
		"""
		Args:
			window_size (int): Window size for the input sequences
			batch_size (int): batch size for the dataset
			target_size (int): target size for the output sequences
			shuffle_buffer (int): buffer size for shuffling the dataset
		"""

		self._window_size = window_size
		self._batch_size = batch_size
		self._target_size = target_size
		self._shuffle_buffer = shuffle_buffer

	def fit(self, X, y=None):
		return self
			
	def transform(self, X: pd.Series) -> tf.data.Dataset:
		series = tf.expand_dims(X, axis=-1)
		dataset = tf.data.Dataset.from_tensor_slices(series)
		dataset = dataset.window(self._window_size+self._target_size, shift=1, drop_remainder=True)
		dataset = dataset.flat_map(lambda window : window.batch(self._window_size+self._target_size))
		if self._shuffle_buffer:
			dataest = dataset.shuffle(self._shuffle_buffer)

		dataset = dataset.map(lambda window: (window[:-self._target_size], window[-self._target_size:]))
		dataset = dataset.batch(self._batch_size).prefetch(1)
		return dataset

class Loader:
	def __init__(self):
		pass
	def load_scaler_from_artifact(self, artifact):
		artifact_dir = artifact.download()
		files_list = os.listdir(artifact_dir)
		scaler_path = f"{artifact_dir}/{files_list[files_list.index('scaler.pkl')]}"

		def load_scaler(scaler_path):
			with open(scaler_path, 'rb') as f:
				scaler = pickle.load(f)
				return scaler

		scaler = load_scaler(scaler_path)
		return scaler

class DataProcessor:
	"""Pipeline and Implementation of Processing time series dataset
	"""
	def __init__(self, config: ProjectConfig):
		self._config = config
		self.config_ds = self._config.dataset
		self.project_name = self._config.project_name
		self.general_config = self._config.general
		self.process_id = get_dt_now()
	
	def prepare_data(self, data: pd.DataFrame) -> tuple:
		"""Pipeline prepare the data to be ready to preprocess for training

		Args:
			data (pd.DataFrame): Original dataframe of USD/IDR time series data

		Returns:
			tuple: (X_train, X_test) - Splitted train and test dataframe
		"""
		class Remover(BaseEstimator, TransformerMixin):
			def fit(self, X: pd.DataFrame, y=None):
				return self
			def transform(self, X: pd.DataFrame) -> pd.DataFrame:
				X.drop(columns=['Volume', 'Dividends','Stock Splits'], inplace=True)
				X_removed = X[
					(X['Close'] > 5000) & (X['Close'] < 30000) &
					(X['Open'] > 5000) & (X['Open'] < 30000) &
					(X['High'] > 5000) & (X['High'] < 30000) &
					(X['Low'] > 5000) & (X['High'] < 30000)
				]
				return X_removed
		
		class Splitter(BaseEstimator, TransformerMixin):
			def __init__(
				self, 
				config: ProjectConfig,
				test_fraction: float, 
				regist_resuts: bool = True
			):
				self._config = config
				self._regist_resuts = regist_resuts
				self._test_fraction = test_fraction

				self.config_ds = self._config.dataset
				self.config_split = {}
			
			def fit(self, X, y=None):
				return self

			def transform(self, X) -> pd.DataFrame:
				split_index = int(len(X) * (1 - self._test_fraction))
				X_train = X.iloc[:split_index]
				X_test = X.iloc[split_index:]

				self.config_split['train_size'] = len(X_train)
				self.config_split['test_size'] = len(X_test)

				if self._regist_resuts:
					self._register_to_wandb(X_train, X_test)

				return X_train, X_test

			def _register_to_wandb(self, X_train, X_test):
				split_dir = os.path.join(
					PROJECT_WORKING_DIR,
					'datasets',
					'splitted_datasets'
				)

				if os.path.exists(split_dir):
					shutil.rmtree(split_dir)
				os.makedirs(split_dir)

				saved_ds_train_path = f'{split_dir}/train_ds_{self.config_ds["start_date"]}_{self.config_ds["end_date"]}.csv'
				saved_ds_valid_path = f'{split_dir}/valid_ds_{self.config_ds["start_date"]}_{self.config_ds["end_date"]}.csv'

				# Save splitted datasets to local directory
				X_train.to_csv(saved_ds_train_path)
				X_test.to_csv(saved_ds_valid_path)

				with wandb.init(
					project=self._config.project_name,
					job_type=f'add-train-valid-set-{get_dt_now()}'
				) as run:
					artifact = wandb.Artifact(
						name='splitted_dataset',
						type='dataset',
						metadata=self.config_split
					)

					artifact.add_dir(split_dir)
					run.log_artifact(artifact)
					run.finish()
			

		pipeline = Pipeline(steps=[
			('remover', Remover()),
			('splitter', Splitter(
				test_fraction=self.config_ds['test_fraction'],
				config=self._config
				)
			)
		])
		
		X_train, X_test = pipeline.fit_transform(data)
		logger.success("Preparing data successfully")
		return (X_train, X_test)
	
	def preprocess(
		self, 
		X: pd.Series,
		batch_size: Union[8, 16, 32],
		scaler_obj=None, 
		for_inference: bool = False
	) -> tf.data.Dataset:
		"""
		Preprocess single time series data
		use `preprocess_with_registry` to preprocess all kind of batch data with registry to wandb

		Args:
			X (pd.Series): _description_
			batch_size (Union[8, 16, 32]): batch size
			scaler_obj (_type_, optional): _description_. Defaults to None.
			for_inference (bool, optional): _description_. Defaults to False.

		Returns:
			tf.data.Dataset: _description_
		"""

		if scaler_obj == None:
			logger.warning('Using new-fitted scaler object for preprocessing data | تحذيرا')
		scaler = Scaler(scaler_obj=scaler_obj)

		shuffle_buffer = self.general_config['shuffle_buffer_size']
		windower = Windower(
			window_size=self.general_config['windowing_size'],
			batch_size=batch_size,
			target_size=self.general_config['target_size'],
			shuffle_buffer=shuffle_buffer if for_inference else None
		)

		pipeline = Pipeline(steps=[
			('normalization', scaler),
			('Windower', windower)
		])

		if for_inference:
			return pipeline.fit_transform(X)

		preprocessed = pipeline.transform(X) if scaler_obj else pipeline.fit_transform(X)
		return preprocessed
	
	def preprocess_with_registry(
		self,
		raw_artifact_name: str,
        scaler_obj=None,
		version: str = 'latest'
	):
		"""
		Preprocess splitted data that has been stored in wandb, 
		with registry pipeline to wandb dataset artifact

		Args:
			raw_artifact_name (str): Wandb Artifact name that stores splitted dataset
			scaler_obj (_type_, optional): File object scaler. Defaults to None.
			version (str): Version name of artifact
		"""

		with wandb.init(
			project=self.project_name,
			group='data_preprocessing',
			job_type=f'data-preprocessing_{self.process_id}'
		) as run:
			config = self.general_config
			artifact = run.use_artifact(
				raw_artifact_name + ':latest', 
				type='dataset'
			)

			# make a preprocssed dataset directory
			saved_dir = os.path.join(
				PROJECT_WORKING_DIR,
				'datasets',
				'preprocessed'
			)
			if os.path.exists(saved_dir):
				shutil.rmtree(saved_dir)
			os.makedirs(saved_dir)

			# download the raw dataset artifact
			artifact_dir = artifact.download()

			# Define train and valid dataset name
			train_files = f"train_ds_{self.config_ds['start_date']}_{self.config_ds['end_date']}.csv"
			valid_files = f"valid_ds_{self.config_ds['start_date']}_{self.config_ds['end_date']}.csv"

			# load data from artifact
			train_ds = pd.read_csv(f'{artifact_dir}/{train_files}')
			valid_ds = pd.read_csv(f'{artifact_dir}/{valid_files}')
			# train_inputs = np.expand_dims(train_ds['Close'].to_numpy(), axis=-1)
			# valid_inputs = np.expand_dims(valid_ds['Close'].to_numpy(), axis=-1)
			train_inputs = train_ds['Close']
			valid_inputs = valid_ds['Close']

			# run preprocessing pipeline
			config, scaler = self._preprocess_all_batched_data(
				train_inputs=train_inputs,
				valid_inputs=valid_inputs,
				scaler_obj=scaler_obj,
				config=config,
				saved_dir=saved_dir
			)

			# initialize preprocessed data artifact to store assets
			preproc_artifact = wandb.Artifact(
				name=f'preprocessed_data',
				type='dataset',
				metadata=config
			)
			
				
			scaler_path = f'{saved_dir}/scaler.pkl'
			with open(scaler_path, 'wb') as f:
				pickle.dump(scaler, f)
				print(f'Successfully saving scaler file to {scaler_path}')
			
			# add the processed file to wandb artifact
			preproc_artifact.add_dir(local_path=saved_dir)
			# add scaler file to wandb artifact
			preproc_artifact.add_file(local_path=scaler_path)
			
			run.log_artifact(artifact)
			run.log_artifact(preproc_artifact)
			run.finish()
	
	def _preprocess_all_batched_data(
		self,
		train_inputs,
		valid_inputs,
		config,
		saved_dir,
		scaler_obj=None,
	) -> tuple:
		"""Register and preprocss dataset with variant of batch size to local directory

		Args:
			train_inputs (ndarray): Input training data with shape (num_samples, 1)
			valid_inputs (ndarray): Input validation data with shape (num_samples, 1)
			scaler: Scaler object used for data scaling
			config (dict): Configuration
			saved_dir (str): Directory to save preprocessed data

		Returns:
			dict: dataset configuration with updated dataset specs
		"""
		# iterate process for different batch_size
		logger.info('using inputed scaler')
		scaler = Scaler(scaler_obj=scaler_obj)

		for process_num in range(len(config['batch_size'])):
            # 1. scaler process
			train_inputs_scaled = scaler.fit_transform(train_inputs)
			valid_inputs_scaled = scaler.transform(valid_inputs)

            # 2. Windower process
			train_windower = Windower(
                window_size=config['windowing_size'],
                target_size=config['target_size'],
                batch_size=config['batch_size'][process_num],
				shuffle_buffer=config['shuffle_buffer_size']
            )
			valid_windower = Windower(
				window_size=config['windowing_size'],
                target_size=config['target_size'],
                batch_size=config['batch_size'][process_num],
                shuffle_buffer=None
            )

			train_windowed_scaled = train_windower.fit_transform(train_inputs_scaled)
			valid_windowed_scaled = valid_windower.fit_transform(valid_inputs_scaled)

            # Update configuration values with dataset specs
			train_specs = [e for e in train_windowed_scaled.take(1)]
			valid_specs = [e for e in valid_windowed_scaled.take(1)]
			new_config_from_ds_spec = {
                f'batch@{config["batch_size"][process_num]}_train_specs': {
                    'dtype': train_specs[0][0].dtype.name,
                    'shape': [dim for dim in train_specs[0][0].shape],
                    'content': train_specs[0][0].numpy().tolist()
                },
                f'batch@{config["batch_size"][process_num]}_train@target_specs': {
                    'dtype': train_specs[0][1].dtype.name,
                    'shape': [dim for dim in train_specs[0][1].shape],
                    'content': train_specs[0][1].numpy().tolist()
                },

                f'batch@{config["batch_size"][process_num]}_valid_specs': {
                    'dtype': valid_specs[0][0].dtype.name,
                    'shape': [dim for dim in valid_specs[0][0].shape],
                    'content': valid_specs[0][0].numpy().tolist()
                },
                f'batch@{config["batch_size"][process_num]}_valid@target_specs': {
                    'dtype': valid_specs[0][1].dtype.name,
                    'shape': [dim for dim in valid_specs[0][1].shape],
                    'content': valid_specs[0][1].numpy().tolist()
                },
            }
            # update config with a new dataset config
			config.update(new_config_from_ds_spec)

            # save the processed Data to local dir
			train_preproc_file_name = f'train@preprocessed-ds_{self.process_id}_batch@{config["batch_size"][process_num]}'
			valid_preproc_file_name = f'valid@preprocessed-ds_{self.process_id}_batch@{config["batch_size"][process_num]}'
			train_preproc_saved_path = f'{saved_dir}/{train_preproc_file_name}'
			valid_preproc_saved_path = f'{saved_dir}/{valid_preproc_file_name}'
			train_windowed_scaled.save(train_preproc_saved_path)
			valid_windowed_scaled.save(valid_preproc_saved_path)

		return config, scaler
	
	
