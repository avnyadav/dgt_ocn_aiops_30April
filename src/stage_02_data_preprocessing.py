from app_exception import AppException
from app_configuration.configuration import AppConfiguration
from app_logger.logger import logging, log_function_signature
import os, sys
from app_entity.config_entity import PreprocessingConfig
from app_entity.entity import DataIngestionEntity, DataPreprocessingEntity, ExperimentEntity
from collections import namedtuple
import tensorflow as tf


class DataPreprocessing:

    @log_function_signature
    def __init__(self, data_ingestion: DataIngestionEntity, experiment: ExperimentEntity, app_config: AppConfiguration):
        try:

            logging.info("Reading the dataset configuration.")
            preprocessing_config = app_config.get_preprocessing_configuration()
            logging.info(f"Dataset configuration :\n{preprocessing_config}\n read successfully")
            self.data_ingestion = data_ingestion
            self.data_preprocessing = DataPreprocessingEntity(
                experiment_id=experiment.experiment_id,
                preprocessing_config=preprocessing_config
            )
            self.data_preprocessing.status = True
            self.data_preprocessing.message = "Data Preprocessing is initialized."
        except Exception as e:
            self.data_preprocessing.status = False
            self.data_preprocessing.message = f"{self.data_preprocessing.message}\n{e}"
            raise AppException(e, sys) from e

    @log_function_signature
    def get_text_encoder(self) -> DataPreprocessingEntity:
        try:
            encoder = tf.keras.layers.TextVectorization(
                max_tokens=self.data_preprocessing.preprocessing_config.vocal_size)
            self.data_preprocessing.encoder = encoder
            self.data_preprocessing.status = True
            self.data_preprocessing.message = f"{self.data_preprocessing.message}\nEncoder object has been initialized."
            return self.data_preprocessing
        except Exception as e:
            self.data_preprocessing.status = False
            self.data_preprocessing.message = f"{self.data_preprocessing.message}\n{e}"
            raise AppException(e, sys) from e
