import os, sys
from app_exception import AppException
from app_utils.util import read_yaml_file
from collections import namedtuple
from app_logger import logging, log_function_signature
from app_entity.config_entity import DatasetConfig, PreprocessingConfig, TrainingPipelineConfig
from app_entity.config_entity import ModelTrainingConfig

ROOT_DIR = os.getcwd()
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_FILE_NAME)

SCHEMA_FILE_NAME = "dataset_schema.yaml"
SCHEMA_FILE_PATH = os.path.join(ROOT_DIR, SCHEMA_FILE_NAME)

# dataset keys
DATA_SET_KEY = "data_set"
DATA_SET_NAME_KEY = "name"
SCHEMA_KEY = "schema"
BUFFER_SIZE_KEY = "buffer_size"
BATCH_SIZE_KEY = "batch_size"

# preprocessing keys

PREPROCESSING_KEY = "preprocessing"
VOCAB_SIZE_KEY = "vocab_size"

# training configuration keys
TRAINING_CONFIG_KEY = "train_config"
TRAINING_MODEL_ROOT_DIR_KEY = "model_root_dir"
TRAINING_MODEL_SAVE_DIR_KEY = "model_save_dir"
TRAINING_MODEL_CHECKPOINT_DIR_KEY = "model_checkpoint_dir"
TRAINING_MODEL_EPOCH_KEY = "epoch"
TRAINING_MODEL_TENSORBOARD_LOG_DIR_KEY = "tensorboard_log_dir"
TRAINING_MODEL_BASE_ACCURACY_KEY = "base_accuracy"
TRAINING_MODEL_VALIDATION_STEP_KEY = "validation_step"

# Training pipeline config
TRAINING_PIPELINE_KEY = "training_pipeline_config"
ARTIFACT_DIR_KEY = "artifact_dir"


class AppConfiguration:
    """
        Reads the configuration file and returns the configuration object
        """

    @log_function_signature
    def __init__(self):
        try:

            logging.info("Reading the configuration file.")
            self.config_info = read_yaml_file(yaml_file_path=CONFIG_FILE_PATH)
            self.dataset_schema = read_yaml_file(yaml_file_path=SCHEMA_FILE_PATH)
            logging.info("Configuration file read successfully.")
        except Exception as e:
            raise AppException(e, sys) from e

    @log_function_signature
    def get_dataset_configuration(self) -> DatasetConfig:
        try:
            dataset_config = self.config_info[DATA_SET_KEY]
            logging.info(f"Dataset configuration :\n{dataset_config}\n read successfully.")
            response = DatasetConfig(name=dataset_config[DATA_SET_NAME_KEY],
                                     schema=self.dataset_schema[SCHEMA_KEY],
                                     batch_size=dataset_config[BATCH_SIZE_KEY],
                                     buffer_size=dataset_config[BUFFER_SIZE_KEY]
                                     )
            logging.info(f"Dataset configuration: [{response}]")
            return response
        except Exception as e:
            raise AppException(e, sys) from e


if __name__=="__main__":
    AppConfiguration().get_dataset_configuration()