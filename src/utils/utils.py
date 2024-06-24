# Importation des modules
# Modules de base
from typing import Union

import numpy as np
import pandas as pd
# Modules de sklearn
from sklearn.base import BaseEstimator
# Métriques
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline


# Fonction d'entraînement et de prédiction
def fit_and_predict(
    estimator: Union[BaseEstimator, Pipeline],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.array],
    X_test: Union[pd.DataFrame, np.ndarray],
) -> Union[pd.Series, np.array, np.ndarray]:
    """
    Trains an estimator on the provided training data and predicts the labels for the test data.

    Parameters:
        estimator (Union[BaseEstimator, Pipeline]): The machine learning model or pipeline to be trained.
        X_train (Union[pd.DataFrame, np.ndarray]): The training input samples.
        y_train (Union[pd.Series, np.array]): The target values (class labels) for training.
        X_test (Union[pd.DataFrame, np.ndarray]): The input samples for testing.

    Returns:
        Union[pd.Series, np.array, np.ndarray]: The predicted labels for the test data.
    """
    # Entraînement de l'estimateur
    estimator.fit(X=X_train, y=y_train)
    # Prédiction
    y_pred = estimator.predict(X=X_test)

    return y_pred


# Fonction d'évaluation de classifieurs
def evaluate_categorical_predictions(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    average: str = "micro",
) -> pd.Series:
    """
    Evaluates the performance of categorical predictions using common classification metrics.

    Parameters:
        y_true (Union[pd.Series, np.array]): The true labels of the data.
        y_pred (Union[pd.Series, np.array]): The predicted labels by the classifier.
        average (str): The averaging method for multi-class metrics. Default is 'micro'.

    Returns:
        pd.Series: A series containing accuracy, precision, recall, and F1 score.
    """

    return pd.Series(
        [
            accuracy_score(y_true=y_true, y_pred=y_pred),
            precision_score(y_true=y_true, y_pred=y_pred, average=average),
            recall_score(y_true=y_true, y_pred=y_pred, average=average),
            f1_score(y_true=y_true, y_pred=y_pred, average=average),
        ],
        index=["accuracy", "precision", "recall", "f1"],
    )
