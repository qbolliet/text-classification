# Importation des modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
# Modules sklearn
from sklearn.base import BaseEstimator, TransformerMixin
# Modules de NLP
# Preprocessing
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Embeding de mots
from sklearn.feature_extraction.text import TfidfVectorizer
# Embeding de mots + contexte
from gensim.models import Word2Vec

# Téléchargement de la ponctuation et des stopwords
download('punkt')
download('stopwords')
download('wordnet')


# Classe de tokenization de textes
class TokenizerTransformer(TransformerMixin, BaseEstimator):
    """
    A custom transformer that tokenizes and optionally lemmatizes text data in specified columns.

    Parameters:
        text_columns (List[str]): List of column names containing the text data.
        lemmatize (bool): Whether to apply lemmatization to the tokens.

    Methods:
        fit(X, y=None):
            Fits the transformer (no training required for this transformer).

        transform(X, y=None):
            Transforms the text data by tokenizing and optionally lemmatizing it.

        filter_and_tokenize(text):
            Filters and tokenizes the input text, removing stop words and non-alphanumeric tokens.
            Applies lemmatization if specified.
    """

    # Initialisation
    def __init__(self, text_colums : List[str], lemmatize : bool) -> None :
        """
        Initializes the TokenizerTransformer with the specified parameters.

        Parameters:
            text_columns (List[str]): List of column names containing the text data.
            lemmatize (bool): Whether to apply lemmatization to the tokens.
        """
        # Initialisation des paramètres
        self.lemmatize = lemmatize
        self.text_columns = text_colums

    # Méthode d'entraînement du transformer
    def fit(self, X, y=None):
        """
        Fits the transformer (no training required for this transformer).

        Parameters:
            X (pd.DataFrame): The input data containing the text columns.
            y (None): Ignored.

        Returns:
            self: The fitted transformer.
        """
        return self
    
    # Méthode de transformation des données par le transformer
    def transform(self, X, y=None):
        """
        Transforms the text data by tokenizing and optionally lemmatizing it.

        Parameters:
            X (pd.DataFrame): The input data containing the text columns.
            y (None): Ignored.

        Returns:
            pd.DataFrame: The transformed data with tokenized and optionally lemmatized text.
        """
        # Copie indépendante du jeu de données
        X_res = X.copy()
        # Initialisation de tqdm
        tqdm.pandas()
        # Parcours des colonnes de text
        for text_column in self.text_columns :
            X_res[text_column] = X_res[text_column].progress_apply(func=lambda x : self.filter_and_tokenize(text=x))
        
        return X_res
    
    # Méthode de preprocessing des textes
    def filter_and_tokenize(self, text : str):
        """
        Filters and tokenizes the input text, removing stop words and non-alphanumeric tokens.
        Applies lemmatization if specified.

        Parameters:
            text (str): The input text to be tokenized and processed.

        Returns:
            str: The processed text after tokenization and optional lemmatization.
        """
        # Initialisation des stop words
        stop_words = set(stopwords.words('french'))
        
        # Tokenization du texte
        word_tokens = word_tokenize(text.lower())

        # Filtre des stopwords et des token qui ne sont pas des caractères alphanumeriques
        filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]

        # Cas de la lemmatization
        if self.lemmatize :
            # Initialisation du Lemmatizer
            lemmatizer = WordNetLemmatizer()
            # Lemmatisation
            filtered_text = [lemmatizer.lemmatize(word) for word in filtered_text]
        
        return ' '.join(filtered_text)


# Classe implémentant l'embeding TF-IDF
class TFIDFTransformer(TransformerMixin, BaseEstimator) :
    """
    A custom transformer that converts text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) method.

    Parameters:
        text_column (str): The name of the column containing the text data.
        max_features (int): The maximum number of features (terms) to be extracted by the TF-IDF model.

    Methods:
        fit(X, y=None):
            Trains the TF-IDF model on the provided text data.
        
        transform(X, y=None):
            Transforms the text data into numerical features using the trained TF-IDF model.
    """

    # Initialisation
    def __init__(self, text_column : str, max_features : int) -> None :
        """
        Initializes the TFIDFTransformer with the specified parameters.

        Parameters:
            text_column (str): The name of the column containing the text data.
            max_features (int): The maximum number of features (terms) to be extracted by the TF-IDF model.
        """
        # Initialisation des paramètres
        self.text_column = text_column
        self.max_features = max_features

    # Méthode d'entraînement
    def fit(self, X, y=None) :
        """
        Trains the TF-IDF model on the provided text data.

        Parameters:
            X (pd.DataFrame): The input data containing the text column.
            y (None): Ignored.

        Returns:
            self: The fitted transformer.
        """
        # Initialisation de TF-IDF
        self.model = TfidfVectorizer(max_features=self.max_features)
        # Entrainement du model
        self.model.fit(X[self.text_column])

        return self
    
    # Méthode de transformation des données
    def transform(self, X, y=None):
        """
        Transforms the text data into numerical features using the trained TF-IDF model.

        Parameters:
            X (pd.DataFrame): The input data containing the text column.
            y (None): Ignored.

        Returns:
            scipy.sparse.csr.csr_matrix: The transformed data as a sparse matrix of TF-IDF features.
        """
        return self.model.transform(X[self.text_column])


# Classe implémentant l'embeding Word2Vec
class Word2VecTransformer(TransformerMixin, BaseEstimator) :
    """
    A custom transformer that converts text data into numerical features using the Word2Vec model.

    Parameters:
        text_column (str): The name of the column containing the text data.
        num_features (int): The number of features (dimensions) for the Word2Vec embeddings.
        window (int): The maximum distance between the current and predicted word within a sentence.

    Methods:
        fit(X, y=None):
            Trains the Word2Vec model on the provided text data.
        
        transform(X, y=None):
            Transforms the text data into numerical features by averaging Word2Vec embeddings for each sentence.
    """

    # Initialisation
    def __init__(self, text_column : str, num_features : int, window : int) -> None :
        """
        Initializes the Word2VecTransformer with the specified parameters.

        Parameters:
            text_column (str): The name of the column containing the text data.
            num_features (int): The number of features (dimensions) for the Word2Vec embeddings.
            window (int): The maximum distance between the current and predicted word within a sentence.
        """
        # Initialisation des paramètres
        self.text_column = text_column
        self.num_features = num_features
        self.window = window

    # Méthode d'entraînement du transformer
    def fit(self, X, y=None) :
        """
        Trains the Word2Vec model on the provided text data.

        Parameters:
            X (pd.DataFrame): The input data containing the text column.
            y (None): Ignored.

        Returns:
            self: The fitted transformer.
        """
        # Découpage du texte
        sentences = [text.split() for text in X[self.text_column]]
        # Entraînement du Word2Vec
        self.model = Word2Vec(sentences=sentences, vector_size=self.num_features, window=self.window, min_count=1, workers=4)
        
        return self
    
    # Méthode de transformation des données
    def transform(self, X, y=None):
        """
        Transforms the text data into numerical features by averaging Word2Vec embeddings for each sentence.

        Parameters:
            X (pd.DataFrame): The input data containing the text column.
            y (None): Ignored.

        Returns:
            pd.DataFrame: The transformed data with numerical features.
        """
        # Découpage du texte
        sentences = [text.split() for text in X[self.text_column]]
        # Moyennisation des embedings de Word2Vec
        features = np.zeros((len(sentences), self.num_features), dtype="float32")
        # Parcours des phrases
        for i, words in enumerate(sentences):
            feature_vec = np.zeros((self.num_features,), dtype="float32")
            nwords = 0
            # Parcours des mots de chaque phrase
            for word in words:
                if word in self.model.wv.key_to_index:
                    nwords += 1
                    feature_vec = np.add(feature_vec, self.model.wv[word])
            if nwords > 0:
                feature_vec = np.divide(feature_vec, nwords)
            features[i] = feature_vec

        # Conversion en DataFrame
        X_res = pd.concat([X, pd.DataFrame(features, index=X.index, columns=[f"feature_{i}" for i in range(self.num_features)])], axis=1).drop(self.text_column, axis=1)

        return X_res

    