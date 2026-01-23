import os
import sys
import argparse
from dotenv import load_dotenv
from loguru import logger
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.models import ModelBuilder, ModelLoader
from usd_idr_forecasting.trainers import ColdStartTrainer

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")

config_path = os.path.join(PROJECT_WORKING_DIR,'project_configs.yaml')
project_config = ProjectConfig.from_yaml(config_path=config_path)
model_config = project_config.model
project_name = project_config.project_name
wandb_team_name = project_config.wandb_team_name

parser = argparse.ArgumentParser()
parser.add_argument(
	"rnn",
	choices=['lstm', 'gru'],
	help="RNN architectur type name to be used",
)

parser.add_argument(
	"--from-checkpoint",
	action='store_true',
	help='Use last checkpoint model, else use from config yaml file',
)

parser.add_argument(
	"--evaluate-on",
	choices=['train', 'valid', 'both'],
	default=None,
	help='Run evaluate',
)

args = parser.parse_args()
print(args.rnn)
if args.from_checkpoint:
	print('load model from checkpoint')
	model = ModelLoader(config=project_config) \
		.load_model_from_artifact(
			artifact_name=f"{wandb_team_name}/{project_name}/retrained-5best-{args.rnn}:checkpoint",
			rnn_type=args.rnn
		)
	
	logger.success("Successfully loaded model from checkpoint")
else:
	model_builder = ModelBuilder(config=project_config)
	model_builder._build_network(args.rnn)
	model = model_builder.build()
	logger.success("Successfully loaded model from config")


trainer = ColdStartTrainer(
	config=project_config,
	model_config=model_config
)

trainer.start(model=model)

if args.evaluate_on == 'both':
	trainer.evaluate(split_mode='train')
	trainer.evaluate(split_mode='valid')

elif args.evaluate_on == None:
	print('Doesn\'t run any evaluation')
	sys.exit(0)
	
else:
	print(f"run evaluation on {args.evaluate_on}")
	trainer.evaluate(split_mode=args.evaluate_on)
# trainer.evaluate