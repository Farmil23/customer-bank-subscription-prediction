import os
import sys
import pandas as pd
import numpy as np

# Import semua model dan metrik yang dibutuhkan
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from bankmarketing.exception.exception import BankMarketingException
from bankmarketing.logging.logger import logging
from bankmarketing.entity.config_entity import ModelTrainerConfig
from bankmarketing.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from bankmarketing.utils.main_utils.utils import load_numpy_array_data, save_object

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise BankMarketingException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Melatih dan mengevaluasi semua model yang diberikan, lalu mengembalikan laporan performa.
        """
        try:
            report = {}
            for model_name, model in models.items():
                logging.info(f"--- Melatih Model: {model_name} ---")
                model.fit(X_train, y_train) # Melatih model

                # Prediksi pada data training dan testing
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Menghitung metrik F1 Score untuk training dan testing
                train_model_score = f1_score(y_train, y_train_pred)
                test_model_score = f1_score(y_test, y_test_pred)

                # Menyimpan F1 score testing ke dalam laporan
                report[model_name] = test_model_score

                # Mencetak laporan lengkap seperti di notebook-mu
                logging.info(f"Laporan Performa untuk: {model_name}")
                logging.info(f"  Training F1 Score: {train_model_score:.4f}")
                logging.info(f"  Testing F1 Score: {test_model_score:.4f}")
                logging.info(f"  Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
                logging.info(f"  Testing Precision: {precision_score(y_test, y_test_pred):.4f}")
                logging.info(f"  Testing Recall: {recall_score(y_test, y_test_pred):.4f}")
                logging.info("-" * 40)
            
            return report

        except Exception as e:
            raise BankMarketingException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Memulai komponen Model Trainer.")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Daftar model klasifikasi yang akan dievaluasi
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "Adaboost": AdaBoostClassifier(),
                "Xgboost": XGBClassifier()
            }
            
            # Memulai evaluasi model
            model_report: dict = self.evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models
            )
            
            # Mendapatkan model terbaik berdasarkan F1 score tertinggi
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Model terbaik ditemukan: {best_model_name} dengan F1 Score Test: {best_model_score:.4f}")

            # Mengecek apakah model terbaik memenuhi skor minimum
            if best_model_score < self.model_trainer_config.base_accuracy:
                raise Exception(f"Tidak ada model yang cukup baik. Skor terbaik: {best_model_score:.4f}")

            logging.info("Model terbaik lulus pengecekan performa minimum.")
            
            # Menyimpan model terbaik
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Model terbaik berhasil disimpan di: {self.model_trainer_config.trained_model_file_path}")

            # Membuat artifact
            # Menghitung F1 score training dari model terbaik untuk laporan
            best_model.fit(X_train, y_train) # Latih ulang model terbaik dengan data lengkap
            y_train_pred_best = best_model.predict(X_train)
            train_f1_best = f1_score(y_train, y_train_pred_best)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_f1_best,
                test_metric_artifact=best_model_score
            )
            return model_trainer_artifact

        except Exception as e:
            raise BankMarketingException(e, sys)