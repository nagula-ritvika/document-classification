#__author__ = ritvikareddy
#__date__ = 10/7/18


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

MODEL = 'trained-model.joblib'
TFIDF_VECTORIZER = 'tf-idf.pkl'


def train():

    # load data from csv into a pandas data frame
    data = pd.read_csv("shuffled-full-set-hashed.csv.zip", header=None, names=["label", "words"])

    # delete all samples where either the label is missing or the words are missing
    data.dropna(inplace=True)

    # randomly shuffle and split the available training data into 25% test data and 75% train data
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data['words'].values,
                                                                        data['label'].values, test_size=0.25)

    # transform a document into a vector using tf-idf
    tfIdf_vectorizer = TfidfVectorizer()
    X = tfIdf_vectorizer.fit_transform(train_x)

    # save the tf-idf vectorizer so that we can reuse it when we want to transform an actual test sample before
    # classifying it.
    pickle.dump(tfIdf_vectorizer, open(TFIDF_VECTORIZER, 'wb'))

    # since the training data has imbalanced classes, I chose to set the class_weight parameter as balanced so that the
    # classifier selects the class weights based on the distribution of the classes in the training data.
    randomForestClf = RandomForestClassifier(criterion='gini',
                                             n_estimators=50,
                                             min_samples_split=10,
                                             bootstrap=True,
                                             class_weight='balanced')
    randomForestClf.fit(X, train_y)
    y_predicted = randomForestClf.predict(tfIdf_vectorizer.transform(test_x))
    # save trained model
    joblib.dump(randomForestClf, MODEL)
    confusion_matrix = pd.DataFrame(metrics.confusion_matrix(test_y, y_predicted),
                                    columns=randomForestClf.classes_,
                                    index=randomForestClf.classes_)
    save_cm(confusion_matrix)
    print("Training Data Accuracy = ", metrics.accuracy_score(train_y, randomForestClf.predict(X)))
    print("Test Data Accuracy = ", metrics.accuracy_score(test_y, y_predicted))


# save confusion matrix as a heat map and a csv
def save_cm(confusion_matrix):
    confusion_matrix.to_csv('confusion_matrix.csv')
    plt.pcolor(confusion_matrix)
    plt.yticks(np.arange(0.5, len(confusion_matrix.index), 1), confusion_matrix.index)
    plt.xticks(np.arange(0.5, len(confusion_matrix.columns), 1), confusion_matrix.columns)
    plt.savefig('Confusion-Matrix.png')


if __name__ == '__main__':
    train()
