# Importation des modules
import pandas as pd
# Importation des modules sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
# Importation des modules de NLP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# Classe permettant de raffiner l'entraînement d'un modèle de langage
class LLMClassifier(BaseEstimator, ClassifierMixin):

    # Initialisation
    def __init__(self, text_column,  num_labels=3, model_name='camembert-base', export_name='./finetuned_camembert',  epochs=3, batch_size=16, logging_steps=10, warmup_steps=500, weight_decay=0.01):
        self.text_column = text_column
        self.num_labels = num_labels
        self.model_name = model_name
        self.export_name = export_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.logging_steps = logging_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.trainer = None

    # Méthode auxiliaire de tokenisation
    def _tokenize_function(self, examples : dict) -> dict:
        return self.tokenizer(examples[self.text_column], padding="max_length", truncation=True)

    # Méthode de tokenisation des données
    def tokenize(self, data : pd.DataFrame) -> Dataset :
        # Conversion en Hugging Face Datasets
        dataset = Dataset.from_pandas(data)
        # Tokenisation du jeu de données
        dataset = dataset.map(self._tokenize_function, batched=True)
        # Suppression de la colonne de texte
        dataset = dataset.remove_columns([self.text_column])
        # Mise au format d'un tenseur pytorch
        dataset.set_format("torch")

        return dataset

    # Méthode d'entrainement du classifieur
    def fit(self, X, y):
        # Copie indépendante du jeu de données
        train_df = X.copy()
        # Ajout du label
        train_df['label'] = y

        # Tokenisation
        train_dataset = self.tokenize(data=train_df)

        # Définition des arguments d'entraînement
        training_args = TrainingArguments(
            output_dir='./results',          
            num_train_epochs=self.epochs,    
            per_device_train_batch_size=self.batch_size, 
            warmup_steps=self.warmup_steps,  
            weight_decay=self.weight_decay,  
            logging_dir='./logs',
            logging_steps=self.logging_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )

        # Initialisation du Trainer
        self.trainer = Trainer(
            model=self.model,                  
            args=training_args,                 
            train_dataset=train_dataset        
        )

        # Entrainement du modèle
        self.trainer.train()

        # Sauvegarde du modèle entraîné et du tokenizer
        self.model.save_pretrained(self.export_name)
        self.tokenizer.save_pretrained(self.export_name)

        return self

    # Méthode de prédiction
    def predict(self, X):
        # Tokenisation
        test_dataset = self.tokenize(data=X)

        # Prediction en utilisant le modèle entraîné
        predictions = self.trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        
        return preds


# Classe permettant d'effectuer une classification Zero-Shot de textes
