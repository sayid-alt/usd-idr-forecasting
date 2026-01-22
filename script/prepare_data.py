import os
import argparse
import shutil
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.processors import DataProcessor
from usd_idr_forecasting.data.datasets import DatasetLoader

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")

config_path = os.path.join(PROJECT_WORKING_DIR, 'project_configs.yaml')
project_config = ProjectConfig.from_yaml(config_path=config_path)
project_name = project_config.project_name

parser = argparse.ArgumentParser(prog="Data preparation")
group = parser.add_mutually_exclusive_group()
group.add_argument(
	"--from-config",
	action="store_true",
	default=False,
	help="Load sliced data based on config parameter to be prepared",
)
parser.add_argument(
	"-v",
	"--verbose",
	action='store_true'
)
args = parser.parse_args()

dataset_loader = DatasetLoader(config=project_config)

save_dir = os.path.join(
    PROJECT_WORKING_DIR,
    'datasets',
 	'base',
)
if os.path.isdir(save_dir):
	shutil.rmtree(save_dir)
os.makedirs(save_dir)

if args.from_config:
	logger.info(f"Load sliced time series based on {config_path}")
	data = dataset_loader.load_data(
		slicing=True,
		show_viz=True,
	)
else:
	logger.info("Load all time series")
	data = dataset_loader.load_all(
    	show_viz=False,  # If True will show the graph
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