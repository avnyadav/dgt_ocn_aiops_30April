from collections import namedtuple

DatasetConfig = namedtuple("DatasetConfig",
                           ["name", "schema", "buffer_size", "batch_size"])

# Assignment define validation config

PreprocessingConfig = namedtuple("PreprocessingConfig", ["vocal_size"])

ModelTrainingConfig = namedtuple("TrainingConfig", ["model_save_dir",
                                                    "model_checkpoint_dir",
                                                    "model_root_dir",
                                                    "epoch",
                                                    "tensorboard_log_dir",
                                                    "base_accuracy",
                                                    "validation_step",
                                                    ])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])
