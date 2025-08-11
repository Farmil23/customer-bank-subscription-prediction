from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging


## Configuration of the data ingestion config
from bankmarketing.entity.config_entity import DataIngestionConfig

import os
import sys
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split


