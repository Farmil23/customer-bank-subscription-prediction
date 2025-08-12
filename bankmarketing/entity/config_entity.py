from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging

from datetime import datetime
import os
import sys
from bankmarketing.constant import training_pipeline
from dataclasses import dataclass

print(training_pipeline.PIPELINE__NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self, timestamp = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%H_%S")
        self.pipeline_name = training_pipeline.PIPELINE__NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp : str = timestamp
        

class DataIngestionConfig():
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME

# Di dalam file bankmarketing/entity/config_entity.py

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            # Direktori utama untuk semua output dari tahap validasi
            self.data_validation_dir = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
            )

            # Direktori untuk menyimpan data yang dianggap TIDAK VALID
            self.invalid_data_dir = os.path.join(
                self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
            )

            # --- PERBAIKAN DI DUA BARIS DI BAWAH INI ---
            # Simpan file valid langsung di dalam folder 'data_validation'
            self.valid_train_file_path = os.path.join(
                self.data_validation_dir, training_pipeline.TRAIN_FILE_NAME
            )
            self.valid_test_file_path = os.path.join(
                self.data_validation_dir, training_pipeline.TEST_FILE_NAME
            )
            # --- AKHIR PERBAIKAN ---

            # ... sisa kode tetap sama ...
            self.invalid_train_file_path = os.path.join(
                self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME
            )
            self.invalid_test_file_path = os.path.join(
                self.invalid_data_dir, training_pipeline.TEST_FILE_NAME
            )
            
            self.drift_report_file_path = os.path.join(
                self.data_validation_dir,
                training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
            )

            self.schema_file_path = training_pipeline.SCHEMA_FILE_PATH

            self.drift_report_threshold = training_pipeline.DRIFT_REPORT_THRESHOLD

        except Exception as e:
            raise BankMarketingException(e, sys)
        
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            # Direktori utama untuk semua output dari tahap transformasi
            self.data_transformation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, 
                training_pipeline.DATA_TRANSFORMATION_DIR_NAME
            )

            # Path lengkap untuk menyimpan objek preprocessor (misal: preprocessor.pkl)
            self.transformed_object_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
            )

            # Path lengkap untuk menyimpan data training yang sudah ditransformasi dan diseimbangkan
            self.transformed_train_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy") 
            )

            # Path lengkap untuk menyimpan data testing yang sudah ditransformasi
            self.transformed_test_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
            )

        except Exception as e:
            raise BankMarketingException(e, sys)
        
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.trained_model_file_path: str = os.path.join(
                training_pipeline_config.artifact_dir,
                training_pipeline.MODEL_TRAINER_DIR_NAME,
                training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
                training_pipeline.MODEL_FILE_NAME
            )
            self.base_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
            self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        except Exception as e:
            raise BankMarketingException(e, sys)
        
@dataclass
class ModelEvaluationConfig:
    model_evaluation_file_path: str = os.path.join(
        training_pipeline.ARTIFACT_DIR,
        training_pipeline.MODEL_EVALUATION_DIR_NAME,
        training_pipeline.MODEL_EVALUATION_REPORT_NAME
    )
    time_stamp: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")