# Di dalam main.py

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging

# --- PERBAIKI BLOK IMPORT INI ---
# Impor semua Config dari config_entity
from bankmarketing.entity.config_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig, 
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig # <-- Impor dari sini
)

# Impor semua Artifact dari artifact_entity
from bankmarketing.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact # <-- Impor dari sini
)

# Impor semua Komponen dari components
from bankmarketing.components.data_ingestion import DataIngestion
from bankmarketing.components.data_validation import DataValidation
from bankmarketing.components.data_transformation import DataTransformation
from bankmarketing.components.model_trainer import ModelTrainer
from bankmarketing.components.model_evaluation import ModelEvaluation # <-- Impor komponennya saja
# --- AKHIR PERBAIKAN ---

import os, sys
import os, sys

if __name__ == '__main__':
    try:
        # ==============================================================================
        # Inisiasi Konfigurasi
        # ==============================================================================
        training_pipeline_config = TrainingPipelineConfig()

        # ==============================================================================
        # Data Ingestion
        # ==============================================================================
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print("Data Ingestion Artifact:", data_ingestion_artifact)

        # ==============================================================================
        # Data Validation
        # ==============================================================================
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print("Data Validation Artifact:", data_validation_artifact)

        # ==============================================================================
        # Data Transformation
        # ==============================================================================
        logging.info(">>> Memulai Tahap Data Transformation <<<")
        
        # 1. Buat config untuk data transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        
        # 2. Buat objek komponennya
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )

        # 3. Jalankan komponennya
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        logging.info(">>> Tahap Data Transformation Selesai <<<")
        print("Data Transformation Artifact:", data_transformation_artifact)
        
        # ==============================================================================
        # BLOK TERAKHIR: Model Trainer
        # ==============================================================================
         # TAHAP 1: MODEL TRAINER
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        # TAHAP 2: MODEL EVALUATION
        model_evaluation_config = ModelEvaluationConfig(training_pipeline_config)
        model_evaluation = ModelEvaluation(data_transformation_artifact, model_trainer_artifact, model_evaluation_config)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
        
        print("Pipeline Selesai!")
        print(model_evaluation_artifact)
        
    except Exception as e:
        raise BankMarketingException(e, sys)