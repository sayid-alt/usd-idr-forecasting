import os
import argparse
import keras_tuner
from dotenv import load_dotenv
from usd_idr_forecasting.configs import ProjectConfig
from usd_idr_forecasting.models import TemporalHyperModel
from usd_idr_forecasting.tuner import Tuner
from usd_idr_forecasting.data.datasets import DatasetLoader

load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")

config_path = os.path.join(PROJECT_WORKING_DIR, 'project_configs.yaml')
project_config = ProjectConfig.from_yaml(config_path=config_path)

parser = argparse.ArgumentParser(prog="Hyperparameter Tuning")

parser.add_argument(
	"rnn",
	choices=['lstm', 'gru'],
	help="RNN type architecture to be used for tuning"
)

parser.add_argument(
	"--max-trials",
	type=int,
	help="Number of maximum search trial"
)

parser.add_argument(
	"--exec-per-trial",
	type=int,
	help="Number of executions per trial"
)


args = parser.parse_args()

hyper_model = TemporalHyperModel(
	config=project_config,
	model_type=args.rnn
)

tuner = Tuner(
    hypermodel=hyper_model,
    objective=keras_tuner.Objective('val_rmse', direction='min'),
    max_trials=args.max_trials if args.max_trials else 5,
    executions_per_trial=args.exec_per_trial if args.exec_per_trial else 1,
    overwrite=False,
    directory=f".cache/{args.rnn}_tuner",
    project_name=hyper_model.tuner_project_name,
	config=hyper_model._config,
	model_type=hyper_model._model_type,
)

print(tuner.search_space_summary())


train_set, valid_set, _, _ = DatasetLoader(config=project_config) \
	.load_dataset_for_training(batch_size=32)

tuner.start(
	train_set=train_set, 
	valid_set=valid_set
)

logger.success("Successfully Tuning Hyperparameter")
