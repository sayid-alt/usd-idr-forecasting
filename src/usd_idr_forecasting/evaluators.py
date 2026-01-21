import os
import pandas as pd
import wandb
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Union, List
from dotenv import load_dotenv
from usd_idr_forecasting.utils import get_dt_now, wandb_auth, plot_series
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.data.datasets import DatasetLoader
from usd_idr_forecasting.processors import DataProcessor, Loader, Scaler
from usd_idr_forecasting.models import ModelLoader

load_dotenv()
WANDB_IDRX_FORECAST_KEY = os.getenv("WANDB_IDRX_FORECAST_KEY")
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")
wandb_auth(key=WANDB_IDRX_FORECAST_KEY)

class Evaluator:
	def __init__(
		self, 
		config: ProjectConfig, 
		scaler=None
	):
		self._config = config
		self._scaler = scaler
		self.general_config = self._config.general
		self.wandb_team_name = self._config.wandb_team_name
		self.project_name = self._config.project_name
		self.process_id = get_dt_now()
		self.model = None
	
	@property
	def model(self):
		return self._model
	
	@model.setter
	def model(self, model):
		self._model = model

	def on_origin_series(
		self,
		series,
		prep_series,
		save_csv: str = None,
		return_prediction: bool = False
	):
		"""Evaluate forecasting on True series price

		Args:
			model (_type_): Forecasting Model
			series (_type_): True series price
			prep_series (_type_): Preprocessed series to be used on prediction
			scaler (_type_): Scaler file path
			config (dict): Dictionary Configureation
			save_csv (str, optional): Save directory for dataframe prediction vs real price. Defaults to None.

		Returns:
			_type_: _description_
		"""
		if not self.model:
			raise ValueError("Model is not set yet")
		print('prep_sries:', prep_series)
		forecast_scaled = self.model.predict(prep_series)
		forecast = self._scaler.inverse_transform(forecast_scaled)
		print(forecast.shape)

		forecast_df = pd.DataFrame(series['Close'][self.general_config['windowing_size']:])
		forecast_df['Close_Forecast'] = forecast
		if save_csv != None:
			forecast_df.to_csv(save_csv)
		
		if return_prediction:
			return forecast_df, forecast_scaled
		return forecast_df

	def on_test_series(
		self, 
		rnn_type: Union['lstm', 'gru'],
		model_id: int,
		model_rank: int,
		batch_size: Union[8, 16, 32],
		dataset_version: str = 'latest'
	):

		_, _, _, prep_artifact = DatasetLoader(config=self._config) \
			.load_dataset_for_training(batch_size=batch_size)
		self._scaler = Loader().load_scaler_from_artifact(
			artifact=prep_artifact
		)

		test_data_dir = DatasetLoader(config=self._config) \
			.from_wandb(data_term='test', version=dataset_version)
		
		# load model
		artifact_name = f'{self.wandb_team_name}/{self.project_name}/retrained-5best-{rnn_type}:checkpoint'
		model_name = f'{model_id}-{rnn_type}-retrained-best{model_rank}-{batch_size}.keras'
		self.model = ModelLoader(self._config) \
            .load_model_from_artifact(
                artifact_name=artifact_name,
                model_name=model_name,
                rnn_type=rnn_type,
            )

		with wandb.init(
			project=self.project_name, 
			job_type='inference', 
			id=f'inference_{rnn_type}@test_data_{self.process_id}'
		) as run:
			# Load test data
			test_data_path = os.path.join(
				test_data_dir, 
				os.listdir(test_data_dir)[0]
			)
			test_data = pd.read_csv(
				test_data_path, 
				parse_dates=['Date'], 
				index_col='Date'
			)
			test_series = test_data['Close']

			# preparing test data
			test_set = DataProcessor(config=self._config) \
				.preprocess(
					X=test_series,
					batch_size=batch_size,
					scaler_obj=self._scaler,
					for_inference=True
				)
			
			# plot test data Close price
			plot_series(test_data, ['Close'])


			# prediction and then,			
			# save data close and forecast on dataframe
			save_dir = os.path.join(
				PROJECT_WORKING_DIR,
				'datasets',
				'testing'
			)
			if os.path.isdir(save_dir):
				shutil.rmtree(save_dir)
			os.makedirs(save_dir)

			model_pred_path = f'{save_dir}/inference-{rnn_type}@test_data.csv'
			model_pred_df, model_pred = self.on_origin_series(
				series=test_data,
				prep_series=test_set, 
				save_csv=model_pred_path,
				return_prediction=True
			)

			# save the close forecast dataframe to wandb artifact
			save_artifact = wandb.Artifact(
				name=f'inference-{rnn_type}-test_data',
				type='dataset',
				metadata={    
					'model_id': model_id,
					'model_rank': model_rank,
					'batch_size': batch_size,
				}
			)

			save_artifact.add(wandb.Table(dataframe=model_pred_df), model_pred_path)
			save_artifact.add_file(model_pred_path)

			test_series_scaled = Scaler(scaler_obj=self._scaler).fit_transform(test_series)

			# save metrics
			rmse = tf.keras.metrics.RootMeanSquaredError()
			rmse.update_state(test_series_scaled[42:], model_pred)
			rmse_score = rmse.result().numpy()

			mse = tf.keras.metrics.MeanSquaredError()
			mse.update_state(test_series_scaled[42:], model_pred)
			mse_score = mse.result().numpy()

			mae = tf.keras.metrics.MeanAbsoluteError()
			mae.update_state(test_series_scaled[42:], model_pred)
			mae_score = mae.result().numpy()

			# mape calculated from the original price (inversed transform) values
			mape = tf.keras.metrics.MeanAbsolutePercentageError()
			mape.update_state(model_pred_df['Close'], model_pred_df['Close_Forecast'])
			mape_score = mape.result().numpy()

			run.log_artifact(save_artifact)
			run.log({
				'rmse': rmse_score,
				'mse': mse_score,
				'mae': mae_score,
				'mape': mape_score
			})
			run.finish()

		# visualize
		plt.figure(figsize=(8, 5))
		plt.plot(model_pred_df['Close'], label='Actual Close Price', marker='o')
		plt.plot(model_pred_df['Close_Forecast'], label='Forecasted Close Price', marker='o')
		plt.grid()
		plt.legend()
		plt.title(f'Actual vs. Forecasted Close Price {rnn_type}-cnn')
		plt.xlabel('Date')
		plt.ylabel('Close Price')
		plt.xticks(rotation=35)
		plt.tight_layout()
		plt.savefig(f"test_forecast_plot_{rnn_type}.png") 
		plt.show()

