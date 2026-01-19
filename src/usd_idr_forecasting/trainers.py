import pickle
import re
import os
import shutil
import wandb
import copy
import tensorflow as tf
import pandas as pd

from wandb.integration.keras import WandbMetricsLogger
from dotenv import load_dotenv
from typing import Union

from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.utils import get_dt_now, wandb_auth
from usd_idr_forecasting.data.datasets import DatasetLoader

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")
WANDB_IDRX_FORECAST_KEY = os.getenv("WANDB_IDRX_FORECAST_KEY")
wandb_auth(key=WANDB_IDRX_FORECAST_KEY)

class ColdStartTrainer:
	def __init__(self, config: ProjectConfig, model_config: dict):
		self._config = config
		self._model_config = model_config
		self.general_config = config.general
		self.project_name = config.project_name
		self.wandb_team_name = config.wandb_team_name
		self.process_id = get_dt_now()
	
	@property
	def get_dataset_bundle(self):
		return self.train, self.valid_set

	def run(self, model, batch_size: int = 32) -> None:
		"""Training for initial params based on the previous paper.

		Args:
			model (keras.src.models.functional.Functional): Keras compiled model
			process_id (str): Unique id for every training process
		
		Results:
			Model history stored in wandb run project
		"""
		# Set global config run model
		self.model_type = model.layers[1].name.lower()

		# Load Dataset for training
		self.train_set, self.valid_set, self.training_batch_size, self.prep_artifact = DatasetLoader(
			self._config).load_dataset_for_training(batch_size=batch_size)
		self.general_config['batch_size'] = batch_size


		# Setting up wandb project
		run =  wandb.init(
			project=self.project_name,
			group='model-training',
			job_type=f'cold-start-{self.model_type}',
			name=f'cold-start-{self.model_type}-training-{self.process_id}',
			config=self._model_config,
			settings=wandb.Settings(init_timeout=120),
			reinit=True
		)
		# initialize model directory for later usage

		# Clear tf backend session
		tf.keras.backend.clear_session()

		# ---> CALLBACKS <---
		wandb_metrics_logger = WandbMetricsLogger()
		early_stopping = tf.keras.callbacks.EarlyStopping(
			monitor='val_loss',
			mode='min',
			patience=10
		)

		# fit and store the log history to variable
		self.model = model
		history = self.model.fit(
			self.train_set,
            epochs=self._model_config['epochs'],
            validation_data=self.valid_set,
            batch_size=batch_size,
            callbacks=[wandb_metrics_logger, early_stopping],
            verbose=2
		)

		# save cold model locally
		models_dir = os.path.join(PROJECT_WORKING_DIR, 'models', 'cold_start_models', self.model_type)
		if os.path.exists(models_dir):
			shutil.rmtree(models_dir)
		os.makedirs(models_dir)
		model_path = f'{models_dir}/model-{self.model_type}:cold-start-{self.process_id}.keras'
		self.model.save(model_path)

		# set Model Metadata
		model_metadata = self._model_config.copy()
		model_metadata['callbacks'] = {
			f'{early_stopping.__class__}': early_stopping.__dict__,
			f'{wandb_metrics_logger.__class__}': wandb_metrics_logger.__dict__
		}
		# store a stable copy of params
		# holds training run parameters (like 'epochs', 'steps', 'samples')
		model_metadata['fit_history'] = copy.deepcopy(history.params)

		# save model to wandb artifact
		model_artifact = wandb.Artifact(
			name=f'cold-start-{self.model_type}',
			type='model',
			metadata=model_metadata
		)

		# add model directory to model artifact
		model_artifact.add_dir(models_dir)
		# log artifact
		run.log_artifact(model_artifact)
		# finish wandb run
		run.finish()
	
	def evaluate(self, split_mode: Union['train', 'valid']):
		if not self.model_type:
			raise ValueError("model_type is not defined. Please run training first or set model_type.")

		loaded_model = self._load_model_from_artifact(artifact_name=f'{self.wandb_team_name}/{self.project_name}/cold-start-{self.model_type}:latest')
		print(loaded_model)
		
		# Create dataset inferenc directory
		df_inference_dir = f'{PROJECT_WORKING_DIR}/datasets/compare'
		if os.path.exists(df_inference_dir):
			shutil.rmtree(df_inference_dir)
		os.makedirs(df_inference_dir)
		
		df_inference_name = f'inference@{self.model_type}:{split_mode}_compare_forecast.csv'
		df_inference_path = f'{df_inference_dir}/{df_inference_name}'

		# Apply inference prediction and return prediction values dataframe
		scaler = self._load_scaler_from_artifact(self.prep_artifact)
		train_series, valid_series = DatasetLoader(self._config).from_wandb(data_term='splits')
		forecast_df = self._compare_forecast_on_df(
			model=loaded_model,
			# real value of series
			series=train_series if split_mode == 'train' else valid_series,
			prep_series=self.train_set if split_mode == 'train' else self.valid_set,  # windowed preprocessed values
			scaler=scaler,
			config=self.general_config,
			save_csv=df_inference_path
		)

		# Upload forecast dataframe to wandb table
		compare_table = wandb.Table(dataframe=forecast_df)

		# initialize dataset artifact
		artifact_dataset = wandb.Artifact(
			name=f'{split_mode}-{self.model_type}--cold-start-compare',
			type='dataset',
		)

		# add comparison inference table to artifact
		artifact_dataset.add(compare_table, f'compare:{split_mode}_forecast_{self.model_type}')
		artifact_dataset.add_file(df_inference_path)

		run = wandb.init(
			project=self.project_name,
			name=f'compare-result-train@{self.model_type}-{self.process_id}',
			job_type='inference',
			group='eval_comparison',
		)

		run.log({"compare": compare_table})
		run.log_artifact(artifact_dataset)
		run.finish()

		return forecast_df

	def _compare_forecast_on_df(self, model, series, prep_series, scaler, config, save_csv: str = None):
		print('prep_sries:', prep_series)
		forecast = model.predict(prep_series)
		forecast = scaler.inverse_transform(forecast)
		print(forecast.shape)
		
		forecast_df = pd.DataFrame(series['Close'][config['windowing_size']:])
		forecast_df['Close_Forecast'] = forecast	
		if save_csv != None:
			forecast_df.to_csv(save_csv)
		return forecast_df
	
	def _load_scaler_from_artifact(self, artifact):
		artifact_dir = artifact.download()
		files_list = os.listdir(artifact_dir)
		scaler_path = f"{artifact_dir}/{files_list[files_list.index('scaler.pkl')]}"

		def load_scaler(scaler_path):
			with open(scaler_path, 'rb') as f:
				scaler = pickle.load(f)
				return scaler

		scaler = load_scaler(scaler_path)
		return scaler
	
	def _load_model_from_artifact(self, artifact_name):
		run = wandb.init(
			project=self.project_name,
			name=f'load_model@{self.model_type}_{self.process_id}',
			job_type='load_model',
			group='loader'
		)
		model_artifact = run.use_artifact(artifact_name, type='model')
		artifact_dir = model_artifact.download()
		run.finish()

		model_name = os.listdir(artifact_dir)[0]
		model_path = f'{artifact_dir}/{model_name}'

		loaded_model = tf.keras.models.load_model(model_path)
		return loaded_model