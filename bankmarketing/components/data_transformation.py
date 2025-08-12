import sys, os
from bankmarketing.logging.logger import logging
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.combine import SMOTETomek

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.constant import training_pipeline
from bankmarketing.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from bankmarketing.entity.config_entity import DataTransformationConfig
from bankmarketing.utils.main_utils.utils import save_numpy_array, save_object, read_yaml_file

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
        Method ini membuat objek preprocessor yang akan menangani semua
        pra-pemrosesan data secara otomatis.
        """
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]

            # Pipeline untuk fitur numerik
            num_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(**training_pipeline.DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ('scaler', StandardScaler())
            ])

            # Pipeline untuk fitur kategorikal
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Kolom numerik: {numerical_columns}")
            logging.info(f"Kolom kategorikal: {categorical_columns}")
            
            # Menggabungkan kedua pipeline menjadi satu preprocessor
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ])
            
            return preprocessor
        except Exception as e:
            raise BankMarketingException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Memulai proses transformasi data.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = self._schema_config["target_column"]
            
            # --- TAMBAHAN KODE UNTUK DROP KOLOM ID ---
            # Kita asumsikan nama kolomnya adalah 'id' sesuai schema,
            # jika berbeda, sesuaikan di sini.
            logging.info("Membuang kolom 'id' dari train dan test dataframe.")
            train_df = train_df.drop('id', axis=1)
            test_df = test_df.drop('id', axis=1)
            # --- AKHIR TAMBAHAN KODE ---
            
            # Memisahkan fitur input dan target untuk data training
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # ... sisa kodenya tetap sama persis ...
            
            # Memisahkan fitur input dan target untuk data testing
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info("Menerapkan objek preprocessor pada data training dan testing.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Menerapkan SMOTETomek untuk menyeimbangkan data training.")
            smt = SMOTETomek(sampling_strategy="auto", random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            # Menggabungkan kembali fitur input dan target menjadi satu array
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Menyimpan objek preprocessor dan array yang sudah ditransformasi
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