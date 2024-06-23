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

    # Initialisation
    def __init__(self, text_colums : List[str], lemmatize : bool) -> None :
        # Initialisation des paramètres
        self.lemmatize = lemmatize
        self.text_columns = text_colums

    # Méthode d'entraînement du transformer
    def fit(self, X, y=None):
        return self
    
    # Méthode de transformation des données par le transformer
    def transform(self, X, y=None):
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

    # Initialisation
    def __init__(self, text_column : str, max_features : int) -> None :
        # Initialisation des paramètres
        self.text_column = text_column
        self.max_features = max_features

    # Méthode d'entraînement
    def fit(self, X, y=None) :
        # Initialisation de TF-IDF
        self.model = TfidfVectorizer(max_features=self.max_features)
        # Entrainement du model
        self.model.fit(X[self.text_column])

        return self
    
    # Méthode de transformation des données
    def transform(self, X, y=None):
        return self.model.transform(X[self.text_column])


# Classe implémentant l'embeding Word2Vec
class Word2VecTransformer(TransformerMixin, BaseEstimator) :

    # Initialisation
    def __init__(self, text_column : str, num_features : int, window : int) -> None :
        # Initialisation des paramètres
        self.text_column = text_column
        self.num_features = num_features
        self.window = window

    # Méthode d'entraînement du transformer
    def fit(self, X, y=None) :
        # Découpage du texte
        sentences = [text.split() for text in X[self.text_column]]
        # Entraînement du Word2Vec
        self.model = Word2Vec(sentences=sentences, vector_size=self.num_features, window=self.window, min_count=1, workers=4)
        
        return self
    
    # Méthode de transformation des données
    def transform(self, X, y=None):
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

    