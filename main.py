from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging

from bankmarketing.entity.config_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig, 
    DataValidationConfig,
    DataTransformationConfig # <-- Tambahkan import ini
)
from bankmarketing.components.data_ingestion import DataIngestion
from bankmarketing.components.data_validation import DataValidation
from bankmarketing.components.data_transformation import DataTransformation # <-- Tambahkan import ini

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

    except Exception as e:
        raise BankMarketingException(e, sys)