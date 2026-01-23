import os
import argparse
import pickle
from dotenv import load_dotenv
from loguru import logger
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.processors import DataProcessor

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")

config_path = os.path.join(
	PROJECT_WORKING_DIR,
	'project_configs.yaml'
)
project_config = ProjectConfig.from_yaml(config_path=config_path)
project_name = project_config.project_name
wandb_team_name = project_config.wandb_team_name

parser = argparse.ArgumentParser(prog='Data Preprocessing')
parser.add_argument(
	"--data-version",
	default='latest',
	help="Splitted Data version stored in wandb artifact",
)
parser.add_argument(
	"--scaler-path",
	help="Object-like scaler path for preprocessing",
)
parser.add_argument(
	"-v", "--verbose",
	action="store_true",
	default=0,
	help="Increase output verbosity",
)
args = parser.parse_args()

# add scaler obj file if given
scaler_obj = None
if args.scaler_path:
	with open(scaler_path, 'rb') as f:
		scaler_obj = pickle.load(f)
		logger.info("Successfully load scaler object")

data_processor = DataProcessor(config=project_config)
data_processor.preprocess_with_registry(
	raw_artifact_name=f"{wandb_team_name}/{project_name}/splitted_dataset",
	scaler_obj=scaler_obj,
	version=args.data_version,
)
logger.info("Preprocessing run successfully")