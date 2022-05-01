from collections import namedtuple
from app_entity.config_entity import DatasetConfig
from app_entity.config_entity import PreprocessingConfig, ModelTrainingConfig

ExperimentEntity = namedtuple("ExperimentEntity", [
    "experiment_id",
    "experiment_name",
    "config_info",
    "experiment_description",
    "execution_start_time_stamp",
    "executed_by_user",
    "executed_by_email",
    "execution_stop_time_stamp",
    "execution_status",
    "execution_description",
    "artifacts_dir",
])


class DataIngestionEntity:
    def __init__(self, experiment_id, train, test, dataset_config: DatasetConfig):
        """

        experiment_id, train, test, dataset_config: DatasetConfig
        """
        self.experiment_id = experiment_id,
        self.train = train
        self.test = test
        self.dataset_config = dataset_config
        self.status = None
        self.message = ""
        self.is_dataset_present = None


class DataValidationEntity:
    # define data validation entity
    pass


class DataPreprocessingEntity:
    def __init__(self, experiment_id, preprocessing_config: PreprocessingConfig):
        self.experiment_id = experiment_id,
        self.encoder = None
        self.preprocessing_config = preprocessing_config
        self.status = None
        self.message = ""


class BestModelEntity:

    def __init__(self, ):
        self.best_model = None
        self.model_path = None
        self.is_best_model_exist = None
        self.status = None
        self.message = ""


class MetricInfoEntity:
    # Assignment define metric to evaluate model performance.
    def __init__(self):
        pass


class TrainedModelEntity:
    def __init__(self, experiment_id, model_training_config: ModelTrainingConfig):
        self.experiment_id = experiment_id
        self.model_training_config = model_training_config
        self.model_architecture = None
        self.model = None
        self.status = None
        self.message = ""
        self.is_model_compiled = False
        self.metric_info = MetricInfoEntity()
        self.history = None


class ModelEvaluationEntity:
    def __init__(self, experiment_id, trained_model: TrainedModelEntity, best_model: BestModelEntity):
        self.experiment_id = experiment_id
        self.trained_model = trained_model
        self.best_model = best_model
        self.is_trained_model_accepted = False
        self.status = None
        self.message = ""


class ExportModelEntity:
    def __init__(self, experiment_id, accepted_model: TrainedModelEntity, export_dir: str):
        self.experiment_id = experiment_id
        self.accepted_model = accepted_model
        self.export_dir = export_dir
        self.status = None
        self.message = ""


class TrainingPipelineEntity:
    def __init__(self,
                 data_ingestion: DataIngestionEntity = None,
                 data_preprocessing: DataValidationEntity = None,
                 model_trainer: TrainedModelEntity = None,
                 ):
        """

        data_ingestion: DataIngestionEntity = None,
        data_preprocessing: DataValidationEntity = None,
        model_trainer: TrainedModelEntity = None,
        """
        self.model_trainer = model_trainer
        self.status = None
        self.message = ""
        self.data_ingestion = data_ingestion
        self.data_preprocessing = data_preprocessing

        #Assigment add data validation, model evaluation and model export model