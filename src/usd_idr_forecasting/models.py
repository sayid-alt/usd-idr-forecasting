import os
import wandb
import keras_tuner
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Union, Type
from loguru import logger
from dotenv import load_dotenv
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError
from tensorflow.keras.layers import (
    GlobalMaxPool1D,
    GlobalAveragePooling1D,
    BatchNormalization,
    LSTM,
    Input,
    Conv1D,
    MaxPooling1D,
    Dense,
    GRU,
    Dropout
)

from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.utils import get_dt_now, wandb_auth

load_dotenv()
WANDB_IDRX_FORECAST_KEY = os.getenv("WANDB_IDRX_FORECAST_KEY")
wandb_auth(key=WANDB_IDRX_FORECAST_KEY)

METRICS = [
            tf.keras.losses.MeanAbsoluteError(name='mae'),
            tf.keras.losses.MeanSquaredError(name='mse'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.losses.MeanAbsolutePercentageError(name='mape')
    	]

class ModelBuilder:
	def __init__(self, config: ProjectConfig):
		self._config = config
		self.general_config = config.general
		self.model_config = config.model
		
		self._model = None
	
	@property
	def config(self):
		return self.model_config
	
	def build(self, summary=True):
		self._model.compile(
			loss=tf.keras.losses.Huber(),
			optimizer=tf.keras.optimizers.RMSprop(
				learning_rate=self.model_config['learning_rate'],
				momentum=self.model_config['momentum']
			),
			metrics=METRICS
		)

		if summary:
			self._model.summary()
		
		return self._model
	
	def _build_rnn_network(
		self,
		rnn_class: Type[Union[tf.keras.layers.LSTM, tf.keras.layers.GRU]]
	):
		config = self.model_config['rnn']
		return rnn_class(**config, name=rnn_class.__name__)

	def _build_network(
		self, 
		model_type: Union['lstm', 'gru']
	) -> None:
	
		input_l = Input(shape=(self.general_config['windowing_size'], 1))
		# LSTM | GRU Layer
		rnn = self._build_rnn_network(LSTM)(input_l) if model_type == 'lstm' else self._build_rnn_network(GRU)(input_l)
		
		# Define model config
		config = self.model_config
		config['rnn_mode'] = 'lstm' if rnn.name == 'lstm_1' else 'gru'

		# CONV1D Layer
		conv1d_config = config['conv1d']
		conv1d = Conv1D(
			filters=conv1d_config['filter'],
			kernel_size=conv1d_config['kernel_size'],
			name=conv1d_config['name'], 
			strides=conv1d_config['strides'], 
			padding=conv1d_config['padding'],
			activation=conv1d_config['activation']
		)(rnn)

		# Flattened layer
		flattened_l = GlobalMaxPool1D()(conv1d)

		# Batch Normalitaion
		batch_norm_1 = BatchNormalization()(flattened_l)

		# DENSE LAYERS
		ffd_config = config['feed_forward_layer']
		## Dense 1
		ffd_dense1_conf = ffd_config['dense_1']
		dense1 = Dense(
			ffd_dense1_conf['dense'], 
			activation=ffd_dense1_conf['activation'], 
			name=ffd_dense1_conf['name']
		)(batch_norm_1)

		## Dropout
		dropout_1 = Dropout(rate=config['dropout'])(dense1)

		## Output
		ffd_output_conf = ffd_config['output']
		output_l = Dense(
			ffd_output_conf['dense'], 
			name=ffd_output_conf['name']
		)(dropout_1)


		# Building network arsitecture model
		self._model = tf.keras.Model(inputs=input_l, outputs=output_l)
		self.model_config = config
		logger.success("Successfully Creating Neural Netowkr Model Architecture")
		logger.info("Access the model with __class__._model or for compiling with __class__.build")




class TemporalHyperModel(keras_tuner.HyperModel):
	def __init__(self, config: ProjectConfig, model_type):
		self._config = config
		self._model_type = model_type

		self.general_config = self._config.general
		self.model_config = self._config.model
		self.tuner_config = self._config.tuner
	
	def _build_temporal_network(
                        self,
                        model_type,
                        rnn_units,
                        conv1d_filters,
                        conv1d_kernel_size,
                        max_pool1d,
                        flattening_layer,
                        batch_norm,
                        dropout,
                        dropout_rate,
                        learning_rate,
                        dense_units,
                        optimizers,
    ):
		input_l = Input(shape=(self.general_config['windowing_size'], 1))
		
        # RNN Layer
		config_params = self.model_config
		if model_type == 'lstm':
			config_params['rnn']['type'] = 'lstm'
			rnn = LSTM(
				rnn_units, 
				return_sequences=True, 
				name='lstm_1',
			)(input_l)
		elif model_type == 'gru':
			config_params['rnn']['type'] = 'gru'
			rnn = GRU(
				rnn_units, 
				return_sequences=True, 
				name='gru_1',
			)(input_l)
		else:
			raise NameError("model_type is str, the value must be either 'lstm' or 'gru'")
    
        # CONV1D Layer
		conv1d_config = config_params['conv1d']
		conv1d = Conv1D(
            filters=conv1d_filters, 
            kernel_size=conv1d_kernel_size,
            name=conv1d_config['name'],
            strides=conv1d_config['strides'],
            padding=conv1d_config['padding'],
            activation=conv1d_config['activation']
        )(rnn)
    
        # if max Pooling layer chosen
		if max_pool1d:
			max_pool1d = MaxPooling1D(config_params['max_pooling_1d'], name='max_pool1d_1')(conv1d)
			# chosen flattening layers
			flattened_l = GlobalAveragePooling1D()(max_pool1d) if flattening_layer == 'GlobalAveragePooling1D' else GlobalMaxPool1D()(max_pool1d)
		else:
			# chosen flattening layers
			flattened_l = GlobalAveragePooling1D()(conv1d) if flattening_layer == 'GlobalAveragePooling1D' else GlobalMaxPool1D()(conv1d)
        
        # add dropout if necessary
		if dropout:
			flattened_l = Dropout(dropout_rate, seed=42)(flattened_l)

        # Choice If applied Batch Normalization
		if batch_norm:
			flattened_l = BatchNormalization()(flattened_l)

		# --- Fast Forward Layers ---
		ff_conf = config_params['feed_forward_layer']
		ff_dense1_conf = ff_conf['dense_1']
        
        # Dense layer
		dense = Dense(
            dense_units,
            activation=ff_dense1_conf['activation'], 
            name=ff_dense1_conf['name']
        )(flattened_l)

        # add dropout if necessary
		if dropout:
			dense = Dropout(dropout_rate, seed=42)(dense)
    
		output_l = Dense(1, name='output')(dense)
    
		model = tf.keras.Model(inputs=input_l, outputs=output_l)

		if optimizers == 'adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		else:
			optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            
		model.compile(
			loss=tf.keras.losses.Huber(),
			optimizer=optimizer,
			metrics=METRICS
		)
		return model


	def build(self, hp):
		model_type = self._model_type
		rnn_units = hp.Int(
			'rnn_units', 
			min_value=self.tuner_config['rnn_units']['min_value'], 
			max_value=self.tuner_config['rnn_units']['max_value'], 
			step=self.tuner_config['rnn_units']['step'],
		)
		conv1d_filters = hp.Int(
			'conv1d_filters', 
			min_value=self.tuner_config['conv1d']['filter']['min_value'], 
			max_value=self.tuner_config['conv1d']['filter']['max_value'], 
			step=self.tuner_config['conv1d']['filter']['step'],
		)
		conv1d_kernel_size = hp.Choice(
			'conv1d_kernel_size', 
			self.tuner_config['conv1d']['kernel_size']
		)
		max_pool1d = hp.Boolean(self.tuner_config['max_pool1d']['name'])
		flattening_layer = hp.Choice(
			'flattening_layer', 
			self.tuner_config['flattening_layer']
		)
		batch_norm = hp.Boolean(self.tuner_config['batch_norm']['name'])
		dropout = hp.Boolean(self.tuner_config['dropout']['name'])
		dropout_rate = hp.Choice(
			"dropout_rate", 
			self.tuner_config['dropout']['rate']
		)
		dense_units = hp.Choice(
			'dense_units', 
			self.tuner_config['dense_units']
		)
		learning_rate = hp.Choice('learning_rate', self.tuner_config['learning_rate'])
		optimizers = hp.Choice('optimizer', self.tuner_config['optimizers'])
        
        
		model = self._build_temporal_network(
                model_type=model_type,
                rnn_units=rnn_units,
                conv1d_filters=conv1d_filters,
                conv1d_kernel_size=conv1d_kernel_size,
                max_pool1d=max_pool1d,
                flattening_layer=flattening_layer,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_rate=dropout_rate,
                dense_units=dense_units,
                learning_rate=learning_rate,
                optimizers=optimizers
        )
		return model
	

class ModelLoader:
	def __init__(self, config: ProjectConfig):
		self._config = config
		self.project_name = self._config.project_name
		self.wandb_team_name = self._config.wandb_team_name
		self.process_id = get_dt_now()
	
	def load_model_from_artifact(
		self, 
		artifact_name: str, 
		rnn_type: Union['lstm', 'gru'],
		model_name: str = None, 
	):
		run = wandb.init(
			project=self.project_name,
			name=f'load_model@{rnn_type}_{self.process_id}',
			job_type='load_model',
			group='loader'
		)
		model_artifact = run.use_artifact(artifact_name, type='model')
		artifact_dir = model_artifact.download()
		run.finish()

		model_name = model_name if model_name else os.listdir(artifact_dir)[0]
		model_path = os.path.join(artifact_dir, model_name)

		loaded_model = tf.keras.models.load_model(model_path)
		return loaded_model

	def load_tuned_model(self, v_lstm, v_gru):
		with wandb.init(
        	project=self.project_name
		) as run:
			# --> LSTM LOAD PROCESS <--
			# retrieve 5 best tuned LSTM
			best5_lstm_artifact = run.use_artifact(
				f'{self.wandb_team_name}/{self.project_name}/model-lstm--tuned-5best:{v_lstm}',
				type='model'
			)
			# get dir, files list, and metadata of best5 lstm
			best5_lstm_dir = best5_lstm_artifact.download()
			best5_lstm_files = os.listdir(best5_lstm_dir)
			best5_lstm_metadata = best5_lstm_artifact.metadata

			# --> GRU LOAD PROCESS <--
			# retrieve 5 best tuned LSTM
			best5_gru_artifact = run.use_artifact(
				f'{self.wandb_team_name}/{self.project_name}/model-gru--tuned-5best:{v_gru}',
				type='model'
			)
			# get dir, files list, and metadata of best5 lstm
			best5_gru_dir = best5_gru_artifact.download()
			best5_gru_files = os.listdir(best5_gru_dir)
			best5_gru_metadata = best5_gru_artifact.metadata

			run.finish()

		print(f'5 best LSTM files:\n{best5_lstm_files}')
		print(f'5 best gru files:\n{best5_gru_files}')

		lstm_loaded = {
			'dir': best5_lstm_dir,
			'files': best5_lstm_files,
			'metadata': best5_lstm_metadata
		}

		gru_loaded = {
			'dir': best5_gru_dir,
			'files': best5_gru_files,
			'metadata': best5_gru_metadata
		}

		return (lstm_loaded, gru_loaded)
	
	def load_retrained_model(
		self, 
		rnn_mode: Union['lstm', 'gru'], 
		version: str,
	):
		with wandb.init(
			project=self.project_name,
			job_type='load_retrained_model'
		) as run:
			model_artifact = run.use_artifact(
				f'{self.wandb_team_name}/{self.project_name}/retrained-5best-{rnn_mode}:{version}', type='model')

			model_dir = model_artifact.download()
			model_metadata = model_artifact.metadata

			run.finish()

			return model_dir, model_metadata
