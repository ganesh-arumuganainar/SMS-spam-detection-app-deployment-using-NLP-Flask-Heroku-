import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.model_selection import GridSearchCV

    

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'bow__max_df': [0.25, 0.5, 0.75, 1.0],
            'bow__max_features': [10, 50, 100, 250, 500, 1000, None],
            'bow__stop_words': ('english', None),
            'tfidf__smooth_idf': (True, False),
            'tfidf__norm': ('l1', 'l2', None),
            }

grid = GridSearchCV(pipeline, parameters, cv=2, verbose=10)

if __name__ == '__main__':
    
    print('IMPORTING DATASET .....')
    data = pd.read_csv('data/SMSSpamCollection', delimiter='\t', names=['label','message'])
    print('DONE .....')

    print('TRAIN TEST SPLIT .....')
    msg_train, msg_test, label_train, label_test = train_test_split(data['message'], data['label'], test_size=0.2)
    print('DONE .....')

    print('TRAINING STARTED .....')
    grid.fit(msg_train, label_train)
    print('TRAINING DONE .....')

    print('RUNNING PREDICTIONS .....')
    predictions = grid.best_estimator_.predict(msg_test)

    print('CLASSIFICATION REPORT .....')
    print(classification_report(predictions,label_test))

    print('CONFUSION MATRIX .....')
    print(confusion_matrix(predictions, label_test))

    print('SAVING THE MODEL .....')
    joblib.dump(grid.best_estimator_, 'saved_model/SMS_spam_model.pkl')

    print('SUCCESSFULLY COMPLETED !')
