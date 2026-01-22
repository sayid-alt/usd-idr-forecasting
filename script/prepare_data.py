import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.processors import DataProcessor
from usd_idr_forecasting.data.datasets import DatasetLoader

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")

config_path = os.path.join(PROJECT_WORKING_DIR, 'project_configs.yaml')
project_config = ProjectConfig.from_yaml(config_path=config_path)
project_name = project_config.project_name

parser = argparse.ArgumentParser(prog="Data preparation")
parser.add_argument(
	"-v",
	"--verbose",
	action='store_true'
)
args = parser.parse_args()


data = DatasetLoader(config=project_config).load_all(
	show_viz=False, # If True will show the graph
	save_path=os.path.join(
		PROJECT_WORKING_DIR,
		'datasets',
		'originals',
		'idrx.csv'
	)
)

data_processor = DataProcessor(config=project_config)
train, test = data_processor.prepare_data(data=data)

if args.verbose:
	print(f"train shape: {train.shape}")
	print(f"test shape: {test.shape}")