import os
import wandb
import shutil
import keras_tuner
import tensorflow as tf

from dotenv import load_dotenv
from typing import Union
from wandb.integration.keras import WandbMetricsLogger

from usd_idr_forecasting.utils import get_dt_now
from usd_idr_forecasting.models import TemporalHyperModel
from usd_idr_forecasting.configs import ProjectConfig


load_dotenv()
PROJECT_WORKING_DIR = os.getenv("PROJECT_WORKING_DIR")


class Tuner(keras_tuner.RandomSearch):
    def __init__(
        self, 
        config: ProjectConfig, 
        model_type: Union["lstm", "gru"],
        process_id: str, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._config = config
        self._model_type = model_type
        self._process_id = process_id
        
        self.general_config = self._config.general
        self.project_name = self._config.project_name
        self.tuner_config = self._config.tuner


    def run_trial(self, trial, *args, **kwargs):
        trial_id = trial.trial_id

        # get tuner name
        tuner_name = self.__class__.__bases__[0].__name__
        
        # set configuration
        config = trial.hyperparameters.values.copy()

        # update config
        config.update(self.general_config)
        config.update({"tuner_name": tuner_name})

        wandb.init(
            project=self.project_name,
            group=f"{self._model_type}-{tuner_name}-keras-tuner",
            job_type=f"{self._model_type}-{tuner_name}_tuner@batch{self.tuner_config['batch_size']}-{self._process_id}",
            name=f"{self._model_type}-{tuner_name}-trial-tuner@batch{self.tuner_config['batch_size']}-{trial_id}",
            tags=[tuner_name, "fine-tuning", f"{self._model_type}", f"batch{self.tuner_config['batch_size']}"],
            config=config
        )

        kwargs["callbacks"] = kwargs.get("callbacks", []) + [WandbMetricsLogger()]

        result = super().run_trial(trial, *args, **kwargs)
        wandb.finish()

        return result
    
    def start(self, train_set, valid_set):
        # run tuning process...
        self.search(train_set, epochs=10, validation_data=valid_set, verbose=2)
        
        # register best result
        self._register_best_models()

        # return 5 best models
        return self.get_best_models(num_models=5)
    
    def _register_best_models(self):
        # get tuner method name
        tuner_name = self.__class__.__bases__[0].__name__

        local_dir = os.path.join(PROJECT_WORKING_DIR, 'models', 'tuned', tuner_name, self._model_type)
        if os.path.isdir(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir)

        # get 5 best hp
        best_lstm_hps = self.get_best_hyperparameters(5)

        # save the 5 models with best hps locally
        for num, hp in enumerate(best_lstm_hps[:5]):
            model_path = f'{local_dir}/model-{self._model_type}:best-tuned-rank{num}.keras'
            model = TemporalHyperModel(config=self._config, model_type=self._model_type).build(hp)
            model.save(model_path)

        # store best 5 tuned LSTM model to wandb
        with wandb.init(project=self.project_name, job_type=f"save-5best-{self._model_type}-model") as run:
            # set artifact metadata
            artifact_metadata = {'hps': [hp.values for hp in best_lstm_hps]}
            # Initialize model artifact
            model_artifact = wandb.Artifact(
                name=f"model-{self._model_type}--tuned-5best",
                type='model',
                metadata=artifact_metadata
            )

            # add local saved LSTM models directory
            model_artifact.add_dir(local_dir)

            # wandb logging
            run.log_artifact(model_artifact)
            run.finish()
    
    
