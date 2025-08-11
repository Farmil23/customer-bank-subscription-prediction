import pandas as pd
import os
import sys
from scipy.stats import ks_2samp

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging
from bankmarketing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from bankmarketing.entity.config_entity import DataValidationConfig
from bankmarketing.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(self.data_validation_config.schema_file_path)
        except Exception as e:
            raise BankMarketingException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # PERBAIKAN: Membandingkan dengan panjang dari key 'columns'
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise BankMarketingException(e, sys)

    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            
            # Cek keberadaan kolom
            missing_numerical_columns = [col for col in numerical_columns if col not in dataframe_columns]
            if len(missing_numerical_columns) > 0:
                logging.warning(f"Missing numerical columns: {missing_numerical_columns}")
                return False

            # Cek tipe data
            for col in numerical_columns:
                if dataframe[col].dtype not in ['int64', 'float64']:
                    logging.warning(f"Column '{col}' is expected to be numerical, but found dtype: {dataframe[col].dtype}")
                    return False
            return True
        except Exception as e:
            raise BankMarketingException(e, sys)

    def validate_categorical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            categorical_columns = self._schema_config["categorical_columns"]
            dataframe_columns = dataframe.columns

            missing_categorical_columns = [col for col in categorical_columns if col not in dataframe_columns]
            if len(missing_categorical_columns) > 0:
                logging.warning(f"Missing categorical columns: {missing_categorical_columns}")
                return False

            for col in categorical_columns:
                if dataframe[col].dtype != 'object':
                    logging.warning(f"Column '{col}' is expected to be categorical (object), but found dtype: {dataframe[col].dtype}")
                    return False
            return True
        except Exception as e:
            raise BankMarketingException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        try:
            drift_report = {}
            # PERBAIKAN: Hanya cek drift pada kolom numerik
            for column in self._schema_config["numerical_columns"]:
                d1 = base_df[column]
                d2 = current_df[column]
                ks_statistic, p_value = ks_2samp(d1, d2)
                
                # PERBAIKAN: Logika p-value yang benar
                if p_value <= self.data_validation_config.drift_report_threshold:
                    is_drifted = True # Drift terdeteksi
                else:
                    is_drifted = False # Tidak ada drift
                
                drift_report[column] = {
                    "p_value": float(p_value),
                    "drift_status": is_drifted
                }
            
            # Simpan laporan drift
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=drift_report)

            # Jika ada satu saja kolom yang drift, anggap seluruh dataset drift
            overall_drift_status = any(d["drift_status"] for d in drift_report.values())
            return overall_drift_status
        except Exception as e:
            raise BankMarketingException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Memulai komponen Data Validation.")
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # 1. Validasi Skema (Jumlah kolom, tipe data numerik, tipe data kategorikal)
            logging.info("Memulai validasi skema data.")
            is_train_schema_valid = all([
                self.validate_number_of_columns(dataframe=train_df),
                self.validate_numerical_columns(dataframe=train_df),
                self.validate_categorical_columns(dataframe=train_df)
            ])
            
            is_test_schema_valid = all([
                self.validate_number_of_columns(dataframe=test_df),
                self.validate_numerical_columns(dataframe=test_df),
                self.validate_categorical_columns(dataframe=test_df)
            ])

            validation_status = is_train_schema_valid and is_test_schema_valid

            if not validation_status:
                error_message = "Validasi skema gagal pada train set atau test set. Periksa log untuk detail."
                logging.error(error_message)
                raise Exception(error_message)
            
            logging.info("Validasi skema berhasil.")

            # 2. Deteksi Data Drift
            logging.info("Memulai deteksi data drift.")
            is_drifted = self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            if is_drifted:
                logging.warning("Data drift terdeteksi antara set training dan testing.")
            else:
                logging.info("Tidak ada data drift yang signifikan.")
            
            # --- TAMBAHAN KODE UNTUK MENYALIN FILE VALID ---
            # 3. Menyalin file yang sudah divalidasi ke direktori 'validated'
            
            # Mengambil path tujuan dari config
            valid_train_path = self.data_validation_config.valid_train_file_path
            valid_test_path = self.data_validation_config.valid_test_file_path

            # Membuat direktori 'validated' jika belum ada
            os.makedirs(os.path.dirname(valid_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(valid_test_path), exist_ok=True)

            # Menyalin file
            train_df.to_csv(valid_train_path, index=False, header=True)
            test_df.to_csv(valid_test_path, index=False, header=True)

            logging.info("Berhasil menyalin file yang sudah divalidasi ke direktori 'validated'.")
            # --- AKHIR TAMBAHAN KODE ---

            # 4. Buat Artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                # PERBAIKAN: Arahkan ke path file yang baru di direktori 'validated'
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data validation artifact berhasil dibuat: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise BankMarketingException(e, sys)