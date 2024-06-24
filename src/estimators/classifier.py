# Importation des modules
import pandas as pd
# Importation des modules sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
# Séparation train/test
from sklearn.model_selection import train_test_split
# Importation des modules de NLP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset


# Classe permettant de raffiner l'entraînement d'un modèle de langage
class LLMClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that fine-tunes a pre-trained language model for sequence classification.

    Parameters:
        text_column (str): The name of the column containing the text data to be classified.
        num_labels (int, optional): The number of labels for classification. Defaults to 3.
        model_name (str, optional): The name of the pre-trained language model to use. Defaults to 'camembert-base'.
        export_name (str, optional): The path to save the fine-tuned model. Defaults to './finetuned_camembert'.
        test_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
        epochs (int, optional): The number of training epochs. Defaults to 3.
        batch_size (int, optional): The batch size for training and evaluation. Defaults to 16.
        logging_steps (int, optional): The number of steps between logging. Defaults to 10.
        warmup_steps (int, optional): The number of warmup steps for learning rate scheduler. Defaults to 500.
        weight_decay (float, optional): The weight decay to apply. Defaults to 0.01.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.
        model (AutoModelForSequenceClassification): The pre-trained language model for sequence classification.
        trainer (Trainer, optional): The Hugging Face Trainer instance for training the model.
    """
    # Initialisation
    def __init__(self, text_column,  num_labels=3, model_name='camembert-base', export_name='./finetuned_camembert', test_size=0.2, epochs=3, batch_size=16, logging_steps=10, warmup_steps=500, weight_decay=0.01):
        self.text_column = text_column
        self.num_labels = num_labels
        self.model_name = model_name
        self.export_name = export_name
        self.test_size = test_size
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
        """
        Tokenizes the input examples using the tokenizer.

        Parameters:
            examples (dict): A dictionary containing the text data to tokenize.

        Returns:
            dict: A dictionary with tokenized data.
        """
        return self.tokenizer(examples[self.text_column], padding="max_length", truncation=True)

    # Méthode de tokenisation des données
    def tokenize(self, data : pd.DataFrame) -> Dataset :
        """
        Tokenizes the input data and prepares it for training.

        Parameters:
            data (pd.DataFrame): The input data to tokenize.

        Returns:
            Dataset: A Hugging Face Dataset with tokenized data.
        """
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
        """
        Fits the classifier by fine-tuning the pre-trained language model.

        Parameters:
            X (pd.DataFrame): The input data containing the text to be classified.
            y (pd.Series): The target labels corresponding to the text data.

        Returns:
            LLMClassifier: The fitted classifier instance.
        """
        # Copie indépendante du jeu de données
        df = X.copy()
        # Ajout du label
        df['label'] = y

        # Séparation du jeu de données en jeu de données d'entraînement et jeu de données de validation
        train_df, val_df = train_test_split(df, test_size=self.test_size, random_state=42)

        # Tokenisation
        train_dataset = self.tokenize(data=train_df)
        eval_dataset = self.tokenize(data=val_df)

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
            train_dataset=train_dataset,
            eval_dataset=eval_dataset     
        )

        # Entrainement du modèle
        self.trainer.train()

        # Sauvegarde du modèle entraîné et du tokenizer
        self.model.save_pretrained(self.export_name)
        self.tokenizer.save_pretrained(self.export_name)

        return self

    # Méthode de prédiction
    def predict(self, X):
        """
        Predicts the labels for the input data using the fine-tuned model.

        Parameters:
            X (pd.DataFrame): The input text data to classify.

        Returns:
            np.array: The predicted labels.
        """
        # Tokenisation
        test_dataset = self.tokenize(data=X)

        # Prédiction en utilisant le modèle entraîné
        predictions = self.trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        
        return preds


# Classe permettant d'effectuer une classification Zero-Shot de textes
class ZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that performs Zero-Shot text classification using pre-trained models.

    Parameters:
        text_column (str): The name of the column containing the text data to be classified.
        zero_shot_model_name (str, optional): The name of the zero-shot classification model to use. Defaults to 'facebook/bart-large-mnli'.
        summarizer_model_name (str, optional): The name of the summarization model to use. Defaults to 'facebook/bart-large-cnn'.
        label_descriptions (dict, optional): A dictionary mapping labels to their descriptions. If None, the descriptions will be generated from the training data. Defaults to None.

    Attributes:
        classifier (pipeline): The Hugging Face pipeline for zero-shot classification.
        summarizer (pipeline): The Hugging Face pipeline for summarization.
        label_descriptions (dict): A dictionary mapping labels to their descriptions.
        reverse_label_mapping (dict): A dictionary mapping descriptions to their corresponding labels.
        candidate_labels (list): A list of candidate label descriptions used for classification.
    """
    def __init__(self, text_column, zero_shot_model_name='facebook/bart-large-mnli', summarizer_model_name="facebook/bart-large-cnn", label_descriptions=None):
        self.text_column = text_column
        self.zero_shot_model_name = zero_shot_model_name
        self.classifier = pipeline("zero-shot-classification", model=self.zero_shot_model_name)
        self.summarizer_model_name = summarizer_model_name
        self.summarizer = pipeline("summarization", model=self.summarizer_model_name)
        self.label_descriptions = label_descriptions
        self.reverse_label_mapping = None
        self.candidate_labels = None

    def fit(self, X, y):
        """
        Fits the classifier by generating or setting the label descriptions and candidate labels.

        Parameters:
            X (pd.DataFrame): The input data containing the text to be classified.
            y (pd.Series): The target labels corresponding to the text data.

        Returns:
            ZeroShotClassifier: The fitted classifier instance.
        """
        # Définition des descriptions possibles des labels
        if self.label_descriptions is None :
            self.label_descriptions = {category : self.generate_summary(text=' '.join(X.loc[y==category, self.text_column])) for category in y.unique()}
        
        # Definition des lables candidats à partir de leur description
        self.candidate_labels = [description for description in self.label_descriptions.values()]

        # Création d'un mapping inverse entre les valeurs numériques et leurs label
        self.reverse_label_mapping = {v: k for k, v in self.label_descriptions.items()}

        return self

    def predict(self, X):
        """
        Predicts the labels for the input data using zero-shot classification.

        Parameters:
            X (pd.Series): The input text data to classify.

        Returns:
            list: The predicted labels.
        """
        # Zero-shot classification
        results = self.classifier(X, candidate_labels=self.candidate_labels, multi_label=False)

        # Association des labels prédits à leur valeur numérique
        predicted_labels = [self.reverse_label_mapping[result['labels'][0]] for result in results]

        return predicted_labels

    def generate_summary(self, text: str, max_length: int = 50, min_length: int = 25) -> str:
        """
        Generates a small description (summary) of the input text.

        Parameters:
        text (str): The input text to summarize.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

        Returns:
        str: The generated summary.
        """
        # Génération du résumé
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

        # Extraction et retour du résumé
        return summary[0]['summary_text']

