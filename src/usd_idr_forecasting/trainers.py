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
from typing import Union, List
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.utils import get_dt_now, wandb_auth
from usd_idr_forecasting.data.datasets import DatasetLoader
from usd_idr_forecasting.models import ModelLoader
from usd_idr_forecasting.evaluators import Evaluator
from usd_idr_forecasting.processors import Loader

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

	def start(
		self, 
		model, 
		batch_size: int = 32
	) -> None:
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
	
	def evaluate(
		self, 
		split_mode: Union['train', 'valid']
	):
		if not self.model_type:
			raise ValueError("model_type is not defined. Please run training first or set model_type.")
		
		# Create dataset inferenc directory
		df_forecast_dir = f'{PROJECT_WORKING_DIR}/datasets/compare'
		if os.path.exists(df_forecast_dir):
			shutil.rmtree(df_forecast_dir)
		os.makedirs(df_forecast_dir)
		
		df_inference_name = f'inference@{self.model_type}:{split_mode}_compare_forecast.csv'
		df_inference_path = f'{df_forecast_dir}/{df_inference_name}'

		# Apply inference prediction and return prediction values dataframe
		scaler = Loader().load_scaler_from_artifact(
			artifact=self.prep_artifact
		)
		train_series, valid_series = DatasetLoader(self._config) \
			.from_wandb(data_term='splits')
		
		evaluator = Evaluator(
			scaler=scaler,
			config=self._config
		)
		evaluator.model = self.model
		forecast_df = evaluator.on_origin_series(
				series=train_series if split_mode == 'train' else valid_series,
				prep_series=self.train_set if split_mode == 'train' else self.valid_set,  # windowed preprocessed values
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

class Retrainer:
	def __init__(
		self, 
		config: ProjectConfig, 
		model_rank_ids: List[int], 
		batch_sizes: List[int], 
		rnn_type: Union['lstm', 'gru']
	):
		self.__is_retrain = False
		self._config = config
		self._model_rank_ids = model_rank_ids
		self._batch_sizes = batch_sizes
		self._rnn_type = rnn_type
		if self._rnn_type not in ['lstm', 'gru']:
			raise ValueError("rnn_type must be either `lstm` or `gru`")
		self.process_id = get_dt_now()
		self.project_name = self._config.project_name
		self.model_config = self._config.model
		self.general_config = self._config.general
		self.retrain_config = {**self.general_config, **self.model_config}
		self.saved_dir = os.path.join(
			f'{PROJECT_WORKING_DIR}', 
			'models', 
			'retrain'
		)
		if os.path.isdir(self.saved_dir):
			shutil.rmtree(self.saved_dir)
		os.makedirs(self.saved_dir)
		self.model_histories = {}
		self.eval_results = {}
	
	@property
	def get_model_histories(self):
		return self.model_histories

	def start(self, callbacks):
		# Initiate process id
		print('retraining process_id:', self.process_id)

		# load 5tuned model for lstm and gru
		lstm_loaded, gru_loaded= ModelLoader(config=self._config) \
			.load_tuned_model(
				v_lstm='latest',
				v_gru='latest'
			)
		
		for bs in self._batch_sizes:
			for rank_model in self._model_rank_ids:
				train_set, valid_set, batch_size, prep_artifact = DatasetLoader(config=self._config) \
					.load_dataset_for_training(batch_size=bs)
				
				# -- Configs --
				self.retrain_config['batch_size'] = batch_size
				self.retrain_config['rnn_type'] = self._rnn_type
				self.retrain_config['hyperparameters'] = lstm_loaded['metadata']['hps'][rank_model] if self._rnn_type == 'lstm' else gru_loaded['metadata']['hps'][rank_model]
				self.retrain_config['callbacks'] = {c.__class__.__name__: vars(c) for c in callbacks}
				self.retrain_config['model_rank_id'] = rank_model

				# -- RETRAINING -- 
				# RNN model directory
				model_rnn_dir = os.path.join(self.saved_dir, f'{self.process_id}-{self._rnn_type}')
				os.makedirs(model_rnn_dir, exist_ok=True)
				
				# set model path for retraining
				best5_dir = lstm_loaded['dir'] if self._rnn_type == 'lstm' else gru_loaded['dir']
				model_name = f'model-{self._rnn_type}:best-tuned-rank{rank_model}.keras'
				model_path = f'{best5_dir}/{model_name}'
				print('Model path:', model_path)
	
				# retraining...
				history = self._retrain(
					model_path, 
					train_set=train_set, 
					valid_set=valid_set, 
					model_save_dir=model_rnn_dir,
					callbacks=callbacks
				)
				# store history
				self.model_histories[f"{bs}_{rank_model}"] = history

		
		# -- POST TRAINING OPERATION --
		with wandb.init(
			project=self.project_name,
			job_type=f'store-best5-retrained-{self._rnn_type}-{self.process_id}'
		) as run:
			# define saved model artifact
			model_artifact = wandb.Artifact(
				name=f'retrained-5best-{self._rnn_type}',
				type='model',
				metadata=self.retrain_config
			)
		
			# add retrained model file to wandb artifact
			model_artifact.add_dir(model_rnn_dir)
		
			# wandb logger
			run.log_artifact(model_artifact)
			run.finish()
		
		# Set variable condition
		self.__is_retrain = True
		
	def _retrain(
			self, 
			model_path, 
			train_set, 
			valid_set,
			model_save_dir,
			callbacks,
	):
			# -- Helper Function --
			def fit_train(model, train_set=train_set, valid_set=valid_set):
				# clear tensorflow backend session
				tf.keras.backend.clear_session()
				# fit retriain model
				model_history = model.fit(
					train_set,
					epochs=50,
					validation_data=valid_set,
					callbacks=callbacks + [WandbMetricsLogger()],
					verbose=2
				)
				return model_history

			# -- MAIN TRAINING OPERATIONS --
						
			# initialize wandb
			with wandb.init(
				project=self.project_name,
				group='model-training',
				id=f'{self.process_id}@{self.retrain_config["rnn_type"]}@b{self.retrain_config["batch_size"]}@{self.retrain_config["model_rank_id"]}',
				job_type=f'{self.process_id}-retrain-best-{self.retrain_config["rnn_type"]}@batch{self.retrain_config["batch_size"]}',
				name=f'retrain_best_hp-{self.retrain_config["rnn_type"]}_@batch{self.retrain_config["batch_size"]}_{self.retrain_config["model_rank_id"]}',
				config=self.retrain_config
			) as run:         

				# -- TRAINING OPERATION --
				model = tf.keras.models.load_model(model_path)
				model.name = f'{self.process_id}-{self.retrain_config["rnn_type"]}-retrained-best{self.retrain_config["model_rank_id"]}-{self.retrain_config["batch_size"]}.keras'
				model_history = fit_train(model)

				# -- POST TRAINING OPERATION --
				# define saved retrain model path
				model_save_path = os.path.join(model_save_dir, model.name)
				
				# save retrain model
				model.save(model_save_path)
				run.finish()

			return model_history
	
	@property
	def get_eval_results(self):
		return self.eval_results
	
	@property
	def is_retrain(self):
		return self.__is_retrain
	
	@is_retrain.setter
	def is_retrain(self, value):
		self.__is_retrain = value

	def evaluate(
		self,
		split_mode: Union['train', 'valid'],
		model_version: str = 'checkpoint'
	):

		'''Logging compare result of true vs pred value into wandb artifact dataset'''
		
		if not self.__is_retrain:
			raise ValueError(f"Retrain process is not run yet. To run, use `__class__.run()` method")

		# load model directory
		model_dir, model_metadata = ModelLoader(config=self._config) \
			.load_retrained_model(
				rnn_mode=self._rnn_type,
				version=model_version
			)
		
		# get split train valid of series
		train_series, valid_series = DatasetLoader(config=self._config)\
			.from_wandb(data_term='splits')

		# initialize dataset artifact
		artifact_dataset = wandb.Artifact(
			name=f'5best_retrained-compare-{self._rnn_type}-{split_mode}',
			type='dataset',
		)

		# Create dataset inference directory
		df_forecast_dir = f'{PROJECT_WORKING_DIR}/datasets/compare-{self._rnn_type}-{self.process_id}' # directory to store in wandb artifact
		if os.path.exists(df_forecast_dir):
			shutil.rmtree(df_forecast_dir)
		os.makedirs(df_forecast_dir)
			
		for batch_size in self._batch_sizes:
			for rank_model_id in self._model_rank_ids:
					# get dataset for training
					train_set, valid_set, batch_size, prep_artifact = DatasetLoader(config=self._config) \
						.load_dataset_for_training(batch_size=batch_size)
					prepared_ds_dir = prep_artifact.download()
					print('prepared dataset files: {}'.format(os.listdir(prepared_ds_dir)))
					self.general_config['batch_size'] = batch_size
				
					# get scaler
					scaler_path = os.path.join(prepared_ds_dir, 'scaler.pkl')
					with open(scaler_path, 'rb') as scaler_file:
							scaler = pickle.load(scaler_file)
			
					# load model
					prefix_model_name = re.search('.*best', os.listdir(model_dir)[0])[0]
					model_name = f'{prefix_model_name}{rank_model_id}-{batch_size}.keras'
					model_path = os.path.join(model_dir, model_name)
					loaded_model = tf.keras.models.load_model(model_path)

					# set inference prediction result directory
					df_inference_name = f'inference@{self._rnn_type}:{split_mode}_forecast_inference_r{rank_model_id}_b{batch_size}.csv'
					df_inference_path = f'{df_forecast_dir}/{df_inference_name}' # for save csv on local
					
					# Apply inference prediction and return prediction values as a dataframe
					if split_mode == 'train':
						series, prep_series = train_series, train_set
					elif split_mode == 'valid':
						series, prep_series = valid_series, valid_set
					else:
						raise ValueError(f'{split_mode} is not available split type')
						
					forecast_df = Evaluator(
						model=loaded_model, 
						scaler=scaler, 
						config=self._config
					).on_origin_series(
							series=series, # real value of series
							prep_series=prep_series, # windowed preprocessed data
							save_csv=df_inference_path
					)

					self.eval_results[f"{batch_size}_{rank_model_id}"] = forecast_df
					
					# Upload forecast dataframe to wandb table
					compare_table = wandb.Table(dataframe=forecast_df)
					artifact_dataset.add(compare_table, df_inference_name)

		# add comparison inference table to artifact
		artifact_dataset.add_dir(df_forecast_dir)
		with wandb.init(
			project=self.project_name,
			name=f'{self.process_id}@{self._rnn_type}-{split_mode}-compare-result',
			group='eval_comparison',
			job_type='inference'
		) as run:
			run.log({'compare': compare_table})
			run.log_artifact(artifact_dataset)
			run.finish()