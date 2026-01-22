import os
import argparse
from dotenv import load_dotenv
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.data.datasets import DatasetLoader

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")

config_path = os.path.join(
	PROJECT_WORKING_DIR,
	'project_configs.yaml'
)
project_config = ProjectConfig.from_yaml(config_path=config_path)

parser = argparse.ArgumentParser(prog='Data Preprocessing')

parser.add_argument(
	"-v", "--verbose",
	action="store_true",
	default=0,
	help="Increase output verbosity",
)
args = parser.parse_args()

dataset_loader = DatasetLoader(config=project_config)
