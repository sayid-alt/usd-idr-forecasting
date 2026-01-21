import pandas as pd
from usd_idr_forecasting.configs import ProjectConfig
class Evaluator:
	def __init__(self, model, scaler, config: ProjectConfig):
		self._model = model
		self._config = config
		self._scaler = scaler
		self.general_config = self._config.general

	def on_origin_series(
		self,
		series,
		prep_series,
		save_csv: str = None
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
		print('prep_sries:', prep_series)
		forecast = self._model.predict(prep_series)
		forecast = self._scaler.inverse_transform(forecast)
		print(forecast.shape)

		forecast_df = pd.DataFrame(
			series['Close'][self.general_config['windowing_size']:])
		forecast_df['Close_Forecast'] = forecast
		if save_csv != None:
			forecast_df.to_csv(save_csv)
		return forecast_df
