from bankmarketing.components.data_ingestion import DataIngestion
from bankmarketing.components.data_validation import DataValidation

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging
from bankmarketing.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig

import os
import sys

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        
        logging.info("Initiate data ingestion")
        
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("data iniation completed")
        print(dataingestionartifact)
        
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initiate Data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)
        
        
        
    except Exception as e:
        raise BankMarketingException(e, sys)