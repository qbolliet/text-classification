# Classifications de textes

Classification des mails des contribuables français adressés à l'administration

## Contenu

Ce répertoire un ensemble de fonctions permettant de classifier les mails des contribuables français.

Il explore 4 méthodes différentes :
- La classification d'un embeding de mots (TD-IDF)
- La classification d'un embeding de mots en tenant compte du contexte de la phrase (Word2Vec)
- La classification d'un embeding de phrases (masked language modelling : BERT)
- La classification zero-shot

Il est organisé comme suite :
- le dossier "src" contient les scripts implémentant les différentes méthodes
- le dosser "notebook" les applique sur un jeu de données d'exemple 

## Utilisation

```bash
git clone <repo_url>
pip install -r requirements.txt
```
