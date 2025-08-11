import os
import sys
import numpy as np
import pandas as pd

### Defining common constant
TARGET_COLUMN = "y"
PIPELINE__NAME: str = "BankMarketing"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME:str = "BankSubsData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
# Menambahkan konstanta baru untuk Data Validation
DRIFT_REPORT_THRESHOLD: float = 0.05
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

### Data ingestion
DATA_INGESTION_COLLECTION_NAME: str = "BankMarketingDataRaw"
DATA_INGESTION_DATABASE_NAME: str = "dbFarmil"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float= 0.2

### Data validation 
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

### Data Schema
