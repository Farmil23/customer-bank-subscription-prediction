from bankmarketing.components.data_ingestion import DataIngestion

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging
from bankmarketing.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
import os
import sys

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        
        logging.info("Initiate data ingestion")
        
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        
    except Exception as e:
        raise BankMarketingException(e, sys)