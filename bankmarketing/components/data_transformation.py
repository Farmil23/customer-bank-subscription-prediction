import os
import sys
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging
from bankmarketing.entity.config_entity import DataTransformationConfig
from bankmarketing.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from bankmarketing.utils.main_utils.utils import read_yaml_file, save_object, save_numpy_array
from bankmarketing.constant import training_pipeline

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
        except Exception as e:
            raise BankMarketingException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Membuat 'mesin' preprocessor (ColumnTransformer) untuk menangani semua
        imputasi, encoding, dan scaling.
        """
        try:
            # Mengambil daftar kolom dari schema
            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]

            # Lini perakitan untuk fitur numerik
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])

            # Lini perakitan untuk fitur kategorikal
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Kolom numerik: {numerical_columns}")
            logging.info(f"Kolom kategorikal: {categorical_columns}")
            
            # Manajer pabrik yang menggabungkan kedua lini perakitan
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ], remainder='passthrough') # 'passthrough' agar fitur baru tidak dibuang
            
            return preprocessor
        except Exception as e:
            raise BankMarketingException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Memulai komponen Data Transformation.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            
            # 1. DROP KOLOM YANG TIDAK RELEVAN
            # Kolom 'id' sudah dibuang di DataIngestion, jadi tidak perlu lagi.
            
            # 2. FEATURE ENGINEERING
            logging.info("Memulai Feature Engineering.")
            for df in [train_df, test_df]:
                df['contact_recency'] = df['pdays'].apply(lambda x: 1 if x != -1 else 0)
                df['is_retired_or_student'] = df['job'].apply(lambda x: 1 if x in ['retired', 'student'] else 0)
            logging.info("Feature Engineering selesai.")

            # 3. MEMBUAT DAN MENERAPKAN PREPROCESSOR
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = self._schema_config["target_column"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Penting: perbarui schema_config di dalam method ini untuk menyertakan fitur baru
            self._schema_config['numerical_columns'].extend(['contact_recency', 'is_retired_or_student'])

            logging.info("Menerapkan objek preprocessor pada data.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # 4. MENANGANI IMBALANCED DATA
            logging.info("Menerapkan SMOTETomek untuk menyeimbangkan data training.")
            smt = SMOTETomek(sampling_strategy="auto", random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            # Menggabungkan kembali array
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 5. MENYIMPAN HASIL (ARTIFACTS)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Menyimpan objek preprocessor dan data yang sudah ditransformasi.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise BankMarketingException(e, sys)