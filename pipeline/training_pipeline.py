from app_entity.entity import ExperimentEntity, TrainingPipelineEntity
from app_logger import logging, log_function_signature, EXPERIMENT_ID

import uuid
from datetime import datetime
from app_configuration.configuration import AppConfiguration
from app_exception import AppException
from src import DataLoader, DataPreprocessing, ModelTrainer
import sys


class TrainingPipeline:

    @log_function_signature
    def __init__(self,
                 experiment_id=None,
                 experiment_name=None,
                 experiment_description=None,
                 execution_start_time_stamp=None,
                 executed_by_user=None,
                 executed_by_email=None,
                 execution_stop_time_stamp=None,
                 execution_status=None,
                 execution_description=None,
                 artifacts_dir=None
                 ):
        try:
            self.app_config = AppConfiguration()
            self.experiment = ExperimentEntity(
                experiment_id=EXPERIMENT_ID if experiment_id is None else experiment_id,
                experiment_name=experiment_name,
                config_info=self.app_config.config_info,
                experiment_description=experiment_description,
                execution_start_time_stamp=datetime.now(),
                executed_by_user=executed_by_user,
                executed_by_email=executed_by_email,
                execution_stop_time_stamp=execution_stop_time_stamp,
                execution_status=execution_status,
                execution_description=execution_description,
                artifacts_dir=artifacts_dir
            )
            self.training_pipeline = TrainingPipelineEntity(data_ingestion=None,
                                                            data_preprocessing=None,
                                                            model_trainer=None
                                                            )

            self.training_pipeline.status = True
            message = "Training pipeline initialized."
            self.training_pipeline.message = f"{self.training_pipeline.message}\n" \
                                             f"{message}"
            logging.info(message)
        except Exception as e:
            self.training_pipeline.status = False
            self.training_pipeline.message = f"{self.training_pipeline.message}\n{e}"
            raise AppException(e, sys) from e

    @log_function_signature
    def start_training(self) -> TrainingPipelineEntity:
        try:
            data_loader = DataLoader(experiment=self.experiment, app_config=self.app_config)

            data_ingestion = data_loader.get_batch_shuffle_dataset()
            self.training_pipeline.data_ingestion = data_ingestion
            data_preprocessor = DataPreprocessing(experiment=self.experiment,
                                                  app_config=self.app_config,
                                                  data_ingestion=data_ingestion
                                                  )
            data_preprocessing = data_preprocessor.get_text_encoder()
            self.training_pipeline.data_preprocessing = data_preprocessor
            model_trainer = ModelTrainer(data_ingestion=data_ingestion,
                                         data_preprocessing=data_preprocessing,
                                         experiment=self.experiment,
                                         app_config=self.app_config
                                         )

            self.training_pipeline.model_trainer = model_trainer.save_model()
            self.training_pipeline.status = True
            self.training_pipeline.message = f"{self.training_pipeline.message}\nModel Created."
            return self.training_pipeline
        except Exception as e:
            raise AppException(e, sys) from e
