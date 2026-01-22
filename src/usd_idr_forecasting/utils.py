import os
import shutill
import pandas as pd
import wandb
import numpy as np
import math
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Union

load_dotenv()
PROJECT_WORKING_DIR = os.getenv('PROJECT_WORKING_DIR')

# TEST
def test_function():
	return "this is a test utils"


# Plot series
def plot_series(df,
                columns,
                start_date=None,
                end_date=None,
                colors=['blue', 'red', 'green', 'brown'],
                fillbar=False,
                title=None
                ):
  """Plots multiple series from a Pandas DataFrame with different colors.

  Args:
    df: Pandas DataFrame containing the time series data.
    columns: A list of column names to plot.
    colors: A list of colors to use for each series.
  """
  # Ensure the DataFrame index is a DatetimeIndex
  if not pd.api.types.is_datetime64_any_dtype(df.index):
    df.apply(lambda x : pd.to_datetime(x.index, format='%Y-%m-%d', utc=True))

  # plot start and end series time if defined, if not, use entire dataframe
  df = df.loc[start_date:end_date] if start_date and end_date else df

  # add columns and rows
  if len(columns) > 1:
    ncols=2
    # adjust rows based on number of columns
    nrows=math.ceil(len(columns) / ncols)
  else:
    ncols=1
    nrows=1

  fig, ax = plt.subplots(figsize=(16, 5*nrows), ncols=ncols, nrows=nrows)
  if len(columns) > 1:
    ax = ax.flatten()
  # Iterate over the columns and plot each one
    for i, column in enumerate(columns):
      if fillbar:
        ax[i].fill_between(df.index, df[column].min(), df[column], alpha=0.7, color=colors[i % len(colors)])
      ax[i].plot(df.index, df[column], color=colors[i % len(colors)], label=column)
      ax[i].set_title(column)
      ax[i].set_xlabel("Time")
      ax[i].set_ylabel("Price")
      ax[i].legend()
      ax[i].grid(True)
  else:
    if fillbar:
      ax.fill_between(df.index, df[columns[0]].min(), df[columns[0]], alpha=0.7)
    ax.plot(df.index, df[columns[0]], color=colors[0], label=columns[0])
    ax.set_title(columns[0])
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

  plt.tight_layout()
  plt.title(title)
  plt.show()


def plot_history(history):
  fig, ax = plt.subplots(figsize=(15, 5), ncols=2, nrows=1)
  print(history.history.keys())

  params = history.history.keys()
  epochs = range(len(history.history['loss']))


  # Iterate over axes objects in the grid
	# Access and iterate over individual axes in the grid
  for i, param in enumerate(['loss', 'mae']):
    ax[i].set_title(param)
    ax[i].plot(epochs, history.history[param], label=param)
    ax[i].plot(epochs, history.history[f'val_{param}'], label=f'val_{param}')
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Loss')
    ax[i].grid(True)
    ax[i].legend()

  plt.show()

def plot_compare(df: pd.core.frame.DataFrame, 
                 columns: list, 
                 title: str = 'Value Comparasion',
                 save_fig: str = None,
                ):

    for col in columns:
        plt.plot(df[col], label=col)
        
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_fig != None:
        plt.savefig(save_fig)
    plt.show()


# Login wandb
def wandb_auth(key):
    if key:
        return wandb.login(
            key=key,
            verify=True,
            relogin=True
        )
    raise ValueError("Input wandb key first!")

def get_dt_now():
    dt = datetime.utcnow()
    dt_now_str = str(dt.year) + str(dt.month) + str(dt.day) + "_" + str(dt.hour+7) + str(dt.minute) + str(dt.second) + "_" + str(dt.microsecond)
    return dt_now_str


# DATA PREPROCESSED UPLOAD
def get_dataset_for_training(batch_size):
    # helper function
    def load_train_valid_data():
        with wandb.init(project='idrx-forecast', job_type='upload-train-valid-data') as run:
            prep_artifact = run.use_artifact('danielteam/idrx-forecast/preprocessed_data:latest', type='dataset')
            prep_artifact_dir = prep_artifact.download()
            prep_files_list = os.listdir(prep_artifact_dir)
            print('available preprocessed ds:\n\t{}'.format(prep_files_list))
        
            run.log_artifact(prep_artifact)
            run.finish()
    
        return prep_artifact, prep_artifact_dir, prep_files_list
    # load all train valid data
    prep_artifact, prep_artifact_dir, prep_files_list = load_train_valid_data()
    
    # set pattern file names
    print(prep_files_list)
    pattern = re.search('@(.*?)@', prep_files_list[1])[0]
    regex_pattern = "{split_mode}{pattern}{batch_size}"
    
    train_file_name = regex_pattern.format(split_mode='train', pattern=pattern, batch_size=batch_size)
    valid_file_name = regex_pattern.format(split_mode='valid', pattern=pattern, batch_size=batch_size)
    
    
    train_index = prep_files_list.index(train_file_name)
    print('train index:', train_index)
    
    valid_index = prep_files_list.index(valid_file_name)
    print('valid index:', valid_index)
    
    train_batch_path = f'{prep_artifact_dir}/{prep_files_list[train_index]}' 
    valid_batch_path = f'{prep_artifact_dir}/{prep_files_list[valid_index]}' 
    
    train_set = tf.data.Dataset.load(train_batch_path)
    valid_set = tf.data.Dataset.load(valid_batch_path)

    return train_set, valid_set, batch_size, prep_artifact


def get_splitted_original_data(version='latest'):
    with wandb.init(
        project='idrx-forecast',
        job_type='upload_splitting_ds'
    ) as run:
        # initialize artifact usage
        split_ds_artifact = run.use_artifact(f'danielteam/idrx-forecast/splitted_dataset:{version}', type='dataset')
        # print metadata of artficat
        print('split metadata:')
        for k, v in split_ds_artifact.metadata.items():
            print(f'{k}: {v}')
    
        # download artifact folder
        split_ds_dir = split_ds_artifact.download()
    
        # logging artifact process
        run.log_artifact(split_ds_artifact)
        run.finish()
    
    
    # define train and test series from downloaded artifact
    print(os.listdir(split_ds_dir))
    split_files_list = os.listdir(split_ds_dir)
    import re
    
    regx = re.search('_(.*01)', split_files_list[0])[0]
    train_name = f"train{regx}.csv"
    valid_name = f"valid{regx}.csv"
    print(train_name)
    
    train_outliers_name = f"train{regx}@outliers.csv"
    valid_outliers_name = f"valid{regx}@outliers.csv"
    print(train_outliers_name)
    
    train_files = split_files_list[split_files_list.index(train_name)]
    valid_files = split_files_list[split_files_list.index(valid_name)]
    
    train_files_outliers = split_files_list[split_files_list.index(train_outliers_name)]
    valid_files_outliers = split_files_list[split_files_list.index(valid_outliers_name)]
    
    train_series = pd.read_csv(f'{split_ds_dir}/{train_files}', index_col='Date', parse_dates=['Date'])
    valid_series = pd.read_csv(f'{split_ds_dir}/{valid_files}', index_col='Date', parse_dates=['Date'])
    train_series_outliers = pd.read_csv(f'{split_ds_dir}/{train_files_outliers}', index_col='Date', parse_dates=['Date'])
    valid_series_outliers = pd.read_csv(f'{split_ds_dir}/{valid_files_outliers}', index_col='Date', parse_dates=['Date'])

    return (train_series, valid_series), (train_series_outliers, valid_series_outliers)


def load_retrained_model(rnn_type, version):
    with wandb.init(
        project='idrx-forecast',
        job_type='load_retrained_model'
    ) as run:
        model_artifact = run.use_artifact(f'danielteam/idrx-forecast/retrained-5best-{rnn_type}:{version}')

        model_dir = model_artifact.download()
        model_metadata = model_artifact.metadata
        
        run.finish()
        
        return model_dir, model_metadata


def log_compare_evaluation_dataframe(
    model_dir,
    split_mode: Union['train', 'valid'],
    model_mode: Union['lstm', 'gru'],
    wandb_init: dict,
    config: dict,
    run_id: str,
    rank_model_ids: list,
    batch_sizes: list,
    inference_type: Union['retrained', 'cold_start']
):
    # Helper Function
    def compare_forecast_on_df(model, series, prep_series, scaler, config, save_csv: str = None):
        '''Create a df of comparison of actual vs forecast'''
        print('start comparing...')
        print('prep_sries:', prep_series)
        
        forecast = model.predict(prep_series)
        forecast = scaler.inverse_transform(forecast)
        print('forecast shape:', forecast.shape)
        
        forecast_df = pd.DataFrame(series['Close'][config['windowing_size']:])
        print('real series shape:', forecast_df.shape[0])
        forecast_df['Close_Forecast'] = forecast
    
        forecast_df.to_csv(save_csv)
        return forecast_df
        
    '''Logging compare result of true vs pred value into wandb artifact dataset'''
    
    # get splitted original data
    series, series_outliers = get_splitted_original_data()
    train_series, valid_series = series # get split train valid of series 
    # initialize dataset artifact
    artifact_dataset = wandb.Artifact(
        name=f'5best_retrained-compare-{model_mode}-{split_mode}',
        type='dataset',
    )

    # Create dataset inference directory
    df_inference_dir = f'{PROJECT_WORKING_DIR}/datasets/compare-{config["rnn_type"]}-{run_id}' # directory to store in wandb artifact
    if os.path.exists(df_inference_dir):
        shutill.rmtree(df_inference_dir)
    os.makedirs(df_inference_dir)
        
    for batch_size in batch_sizes:
        for rank_model_id in rank_model_ids:
                
                # get dataset for training
                train_set, valid_set, batch_size, prep_artifact = get_dataset_for_training(batch_size=batch_size)
                prepared_ds_dir = prep_artifact.download()
                print('prepared dataset files: {}'.format(os.listdir(prepared_ds_dir)))
                config['batch_size'] = batch_size
            
                # get scaler
                scaler_path = os.path.join(prepared_ds_dir, 'scaler.pkl')
                with open(scaler_path, 'rb') as scaler_file:
                        scaler = pickle.load(scaler_file)
        
                # set model path
                if inference_type == 'retrained':
                    prefix_model_name = re.search('.*best', os.listdir(model_dir)[0])[0]
                    model_name = f'{prefix_model_name}{rank_model_id}-{batch_size}.keras'
                    model_path = os.path.join(model_dir, model_name)
        
                elif inference_type == 'cold_start':
                    model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
        
                else:
                    raise ValueError(f"`{inference_type}` is not available inference type")
        
                # load model
                loaded_model = tf.keras.models.load_model(model_path)
        
                df_inference_name = f'inference@{model_mode}:{split_mode}_forecast_inference_r{rank_model_id}_b{batch_size}.csv'
                df_inference_path = f'{df_inference_dir}/{df_inference_name}' # for save csv on local
                
                # Apply inference prediction and return prediction values dataframe
                if split_mode == 'train':
                    series, prep_series = train_series, train_set
                elif split_mode == 'valid':
                    series, prep_series = valid_series, valid_set
                else:
                    raise ValueError(f'{split_mode} is not available split type')
                    
                forecast_df = compare_forecast_on_df(
                    model=loaded_model,
                    series=series, # real value of series
                    prep_series=prep_series, # windowed preprocessed values
                    scaler=scaler,
                    config=config,
                    save_csv=df_inference_path
                )
                
                # Upload forecast dataframe to wandb table
                compare_table = wandb.Table(dataframe=forecast_df)
                artifact_dataset.add(compare_table, df_inference_name)

        
    # add comparison inference table to artifact
    artifact_dataset.add_dir(df_inference_dir)

    with wandb.init(
        project=wandb_init['project'],
        name=wandb_init['name'],
        group=wandb_init['group'],
        job_type=wandb_init['job_type']
    ) as run:
        run.log({'compare': compare_table})
        run.log_artifact(artifact_dataset)
        run.finish()

def inference_pipeline(
    model_dir: str, 
    model_mode: Union['lstm', 'gru'],
    process_id: str, 
    config: dict,
    batch_sizes: list,
    rank_model_ids: list,
    inference_type: Union['retrained', 'cold_start']
):
            
    print('start compare with train data...')
    split_mode = 'train'
    log_compare_evaluation_dataframe(
        model_dir=model_dir,
        model_mode=model_mode,
        split_mode=split_mode,
        config=config,
        wandb_init=dict(
            project='idrx-forecast',
            group='eval_comparison',
            name=f'{process_id}@{model_mode}-{split_mode}-compare-result',
            job_type='inference'
        ),
        run_id=process_id,
        rank_model_ids=rank_model_ids,
        batch_sizes=batch_sizes,
        inference_type=inference_type
    )

    import time
    time.sleep(5)

    print('start compare with valid data')
    split_mode = 'valid'
    log_compare_evaluation_dataframe(
        model_dir=model_dir,
        model_mode=model_mode,
        split_mode=split_mode,
        config=config,
        wandb_init=dict(
            project='idrx-forecast',
            group='eval_comparison',
            name=f'{process_id}@{model_mode}-{split_mode}-compare-result',
            job_type='inference'
        ),
        run_id=process_id,
        rank_model_ids=rank_model_ids,
        batch_sizes=batch_sizes,
        inference_type=inference_type
    )

