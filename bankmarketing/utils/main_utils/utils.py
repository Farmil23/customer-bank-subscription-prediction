import yaml
from bankmarketing.exception.exception import BankMarketingException
import pandas as pd
import os, sys
from bankmarketing.logging.logger import logging
# import dill
import pickle
import numpy as np

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise BankMarketingException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
            
    except Exception as e:
        raise BankMarketingException(e, sys)