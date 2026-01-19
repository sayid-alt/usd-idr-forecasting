
import os
import re
import shutil
import wandb
import yfinance as yf
import pandas as pd
import tensorflow as tf

from typing import Union
from dotenv import load_dotenv
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.utils import plot_series, wandb_auth

load_dotenv()
PROJECT_WORKING_DIR = os.getenv('PROJECT_WORKING_DIR')
WANDB_IDRX_FORECAST_KEY = os.getenv('WANDB_IDRX_FORECAST_KEY')
wandb_auth(key=WANDB_IDRX_FORECAST_KEY)

class DatasetLoader:
	def __init__(self, config: ProjectConfig):
		self._config = config
		self.project_name = config.project_name
		self.wandb_team_name = config.wandb_team_name
		self.config_ds = config.dataset

	def load_all(self, show_viz: bool = False) -> pd.DataFrame:
		"""Load All time USD/IDR data

		Args:
			show_viz (bool, optional): Will show the visualization of the dataset. Defaults to False.

		Returns:
			pd.DataFrame: dataframe of all time USD/IDR data
		"""
		idrx = yf.Ticker("IDR")
		idrx_history = idrx.history(period="max")  # all data of IDR=X
		idrx_history.sort_values(by='Date', ascending=True, inplace=True)

		if show_viz:
			print(idrx_history.head())
			print(idrx_history.info())
			plot_series(idrx_history, columns=idrx_history.columns)

		return idrx_history
	
	def load_sliced_data(
		self, 
		register_to_wandb: bool = True,
		show_viz: bool = True
	) -> pd.DataFrame:
		"""Load sliced time series of USD/IDR data based on start_date and end_date 

		Args:
			register_to_wandb (bool, optional): Will register dataset to wandb project. Defaults to True.
			show_viz (bool, optional): Will show the visualization of the dataset. Defaults to True.

		Returns:
			pd.DataFrame: dataframe of sliced time series USD/IDR data
		"""
		sliced_ds = idrx_history.loc[self.config_ds['start_date']: self.config_ds['end_date']]

		ds_save_file = f'main_ds_{self.config_ds["start_date"]}_{self.config_ds["end_date"]}.csv'
		ds_save_path = os.path.join(
			PROJECT_WORKING_DIR,
			'data',
			ds_save_file
		)

		# saving sliced dataset to local directory
		sliced_ds.to_csv(ds_save_path)

		if register_to_wandb:
			# save and logging dataset
			with wandb.init(
				project=self._config.project_name,
				group='dataset-upload',
				job_type='add-sliced-dataset'
			) as run:
				# initialize dataset artifact for usd-idr-ds
				dataset_metadata = self.config_ds
				dataset_metadata['row_num'] = sliced_ds.shape[0]
				artifact = wandb.Artifact(
					name='usd-idr-ds',
					type='dataset',
					description='A sliced time series dataset of usd/idr price',
					metadata=dataset_metadata
				)

				# add file to an artifact
				artifact.add_file(
					local_path=ds_save_path,
					name=ds_save_file
				)

				# loggin artifact
				run.log_artifact(artifact)
				run.finish()

		if show_viz:
			plot_series(sliced_ds, columns=['Close'])
			
		return sliced_ds
	
	def load_dataset_for_training(self, batch_size: int) -> tuple:
		"""Load package of datasets and wandb artifact for training usage

		Args:
			batch_size (int): Number of batch size of preprocessed datasets

		Returns:
			tuple: tuple of (train_set, valid_set, batch_size, prep_artifact)
		"""
		# helper function
		def load_train_valid_data(project_name=self.project_name):
			with wandb.init(project=project_name, job_type='upload-train-valid-data') as run:
				prep_artifact = run.use_artifact(f'{self.wandb_team_name}/{self.project_name}/preprocessed_data:latest', type='dataset')
				prep_artifact_dir = prep_artifact.download()
				prep_files_list = os.listdir(prep_artifact_dir)
				print('available preprocessed datasets:\n\t{}'.format(prep_files_list))
			
				run.log_artifact(prep_artifact)
				run.finish()
		
			return prep_artifact, prep_artifact_dir, prep_files_list
		# load all train valid data
		prep_artifact, prep_artifact_dir, prep_files_list = load_train_valid_data(project_name=self.project_name)
		
		# set pattern file names
		print(prep_files_list)
		pattern = re.search('@(.*?)@', prep_files_list[0])[0]
		regex_pattern = "{split_mode}{pattern}{batch_size}"
		
		train_file_name = regex_pattern.format(split_mode='train', pattern=pattern, batch_size=batch_size)
		valid_file_name = regex_pattern.format(split_mode='valid', pattern=pattern, batch_size=batch_size)
		
		
		train_index = prep_files_list.index(train_file_name)
		print('train index:', train_index)
		
		valid_index = prep_files_list.index(valid_file_name)
		print('valid index:', valid_index)
		
		train_batch_path = f'{prep_artifact_dir}/{prep_files_list[train_index]}' 
		valid_batch_path = f'{prep_artifact_dir}/{prep_files_list[valid_index]}' 
		
		train_set = tf.data.Dataset.load(train_batch_path)
		valid_set = tf.data.Dataset.load(valid_batch_path)

		return train_set, valid_set, batch_size, prep_artifact
	
	def from_wandb(
		self, 
		data_term: Union['all', 'splits']
	) -> tuple:
		if data_term not in ['all', 'splits']:
			raise ValueError("data_term must be either 'all' or 'splits'")
		
		if data_term == 'all':
			return self._get_all_registered_data()
		
		if data_term == 'splits':
			return self._get_splits_registered_data()
		
	def _get_all_registered_data(self, version='latest') -> pd.DataFrame:
		with wandb.init(
    		project=self.project_name
		) as run:
			main_ds_artifact = run.use_artifact(
				artifact_or_name=f'{self.wandb_team_name}/{self.project_name}/usd-idr-ds:latest',
				type='dataset'
			)

			print('artifact metadata:')
			for k, v in main_ds_artifact.metadata.items():
				print(f'\t- {k}: {v}')
			main_ds_dir = main_ds_artifact.download()
			run.finish()


	def _get_splits_registered_data(self, version='latest') -> tuple:
		"""Private method to get splitted original data

		Args:
			version (str, optional): Version of wandb artifact. Defaults to 'latest'.

		Returns:
			_type_: _description_
		"""

		with wandb.init(
			project=self.project_name,
			job_type='load_splitting_ds'
		) as run:
			# initialize artifact usage
			split_ds_artifact = run.use_artifact(
				f'{self.wandb_team_name}/{self.project_name}/splitted_dataset:{version}', 
				type='dataset'
			)

			# print metadata of artficat
			print('split metadata:')
			for k, v in split_ds_artifact.metadata.items():
				print(f'{k}: {v}')

			# download artifact folder
			split_ds_dir = split_ds_artifact.download()

			# logging artifact process
			run.log_artifact(split_ds_artifact)
			run.finish()

		# Define train and test series from downloaded artifact
		split_files_list = os.listdir(split_ds_dir)
		regx = re.search('_(.*01)', split_files_list[0])[0]

		train_name = f"train{regx}.csv"
		valid_name = f"valid{regx}.csv"
		
		train_files_name = split_files_list[split_files_list.index(train_name)]
		valid_files_name = split_files_list[split_files_list.index(valid_name)]

		train_series = pd.read_csv(
			f'{split_ds_dir}/{train_files_name}', index_col='Date', parse_dates=['Date'])
		valid_series = pd.read_csv(
			f'{split_ds_dir}/{valid_files_name}', index_col='Date', parse_dates=['Date'])

		return (train_series, valid_series)