import pymongo
from dotenv import load_dotenv
import os
from bankmarketing.exception.exception import BankMarketingException

from bankmarketing.logging.logger import logging

# --- 1. Koneksi ke MongoDB ---
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
DATABASE_NAME = "dbFarmil"
COLLECTION_NAME = "BankMarketingDataRaw" # Koleksi yang akan kita proses

try:
    client = pymongo.MongoClient(MONGO_DB_URL)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    logging.info("Berhasil terhubung ke MongoDB._")
    logging.info(f"Jumlah data SEBELUM proses: {collection.count_documents({})}")

except Exception as e:
    logging.info(f"Gagal terhubung ke MongoDB: {e}")
    # Hentikan eksekusi jika gagal konek
    exit()

# --- 2. Definisikan Aggregation Pipeline ---
# Ini adalah 'jurus' utamanya
pipeline = [
    # Tahap 1: Ambil 30.000 dokumen secara acak
    { "$sample": { "size": 30000 } },
    
    # Tahap 2: Ganti (overwrite) koleksi LAMA dengan hasil dari Tahap 1
    { "$out": COLLECTION_NAME }
]


# --- 3. Jalankan Pipeline ---
try:
    logging.info("\nMenjalankan proses sampling dan replace di MongoDB...")
    # Perintah ini tidak mengembalikan dokumen, tapi melakukan aksi di server
    collection.aggregate(pipeline)
    logging.info("Proses berhasil!")

except Exception as e:
    raise BankMarketingException(e, sys)


# --- 4. Verifikasi ---
# Cek kembali jumlah dokumen di koleksi untuk memastikan
logging.info(f"Jumlah data SESUDAH proses: {collection.count_documents({})}")