import os 
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging

class BankDataExtract():
    def __init__(self):
        try:
            # Buat koneksi ke MongoDB sekali saja saat objek dibuat
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            print("Koneksi ke MongoDB berhasil dibuat.")
        except Exception as e:
            raise BankMarketingException(e, sys)
            
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            
            # Menggunakan cara yang lebih direct dan efisien
            records = data.to_dict(orient='records')
            
            return records
        except Exception as e:
            raise BankMarketingException(e, sys)
            
    def insert_data_mongodb(self, records, database_name, collection_name):
        try:
            # Gunakan koneksi yang sudah ada dari __init__
            database = self.mongo_client[database_name]
            collection = database[collection_name] # << INI PERBAIKAN KRITIS
            
            # Hapus data lama di collection agar tidak ada duplikat setiap kali dijalankan
            collection.delete_many({})
            
            # Masukkan data baru
            collection.insert_many(records)
            
            print(f"{len(records)} data berhasil dimasukkan ke collection '{collection_name}' di database '{database_name}'.")
            
            return len(records)
        
        except Exception as e:
            raise BankMarketingException(e, sys)
            
if __name__ == '__main__':
    FILE_PATH = "bank-data/Raw_data.csv" # Pastikan path ini benar
    DATABASE = "dbFarmil"
    COLLECTION = "BankMarketingDataRaw" # Nama collection untuk data mentah
    
    bankobj = BankDataExtract()
    
    records = bankobj.csv_to_json_converter(file_path=FILE_PATH)
    
    # Cetak hanya 5 record pertama untuk verifikasi
    print("Contoh 5 data pertama:")
    print(records[:5])
    
    no_of_records = bankobj.insert_data_mongodb(records, DATABASE, COLLECTION)