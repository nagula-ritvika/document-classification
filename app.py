# __author__ = ritvikareddy
# __date__ = 10/8/18

import logging
import numpy as np
import pickle

from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib

MODEL = 'trained-model.joblib'
TFIDF_VECTORIZER = 'tf-idf.pkl'

app = Flask(__name__)


# check if the given input is empty
def is_input_empty(input_x):
    return input_x == ' ' or input_x == ''


# create json message to handle GET queries
def send_message(confidence, predicted_label):
    data = {'prediction': predicted_label,
            'confidence': str(confidence)+'%'
            }
    resp = jsonify(data)
    resp.status_code = 200

    return resp


# predict the class for input document
def get_predicted_label(input_x):

    # load the saved tf-idf vectorizer
    tfidf_vectorizer = pickle.load(open(TFIDF_VECTORIZER, 'rb'))
    transformed_x = tfidf_vectorizer.transform([input_x])

    # load the trained model
    model = joblib.load(MODEL)
    predicted_label = model.predict(transformed_x)

    predicted_probabilities = model.predict_proba(transformed_x)[0]
    confidence = max(predicted_probabilities)

    target_labels = model.classes_
    probabilities_dict = {label: prob for label, prob in zip(target_labels, predicted_probabilities)}

    return np.round(confidence, decimals=3) * 100, predicted_label[0].title()


@app.route('/', methods=['GET', 'POST'])
def classify():
    logging.info("someone hit the base endpoint")

    if request.method == 'GET':
        if 'words' in request.args:
            input_x = request.args.get('words')
            if is_input_empty(input_x):
                return render_template('input_form.html', empty_input=True)

            confidence, predicted_label = get_predicted_label(input_x)
            resp = send_message(confidence, predicted_label)

            return resp
        else:
            return render_template('input_form.html')

    else:
        input_x = request.form['test_doc']
        if is_input_empty(input_x):
            return render_template('input_form.html', empty_input=True)

        confidence, predicted_label = get_predicted_label(input_x)

        return render_template('input_form.html', predicted_label=predicted_label, confidence=confidence)


if __name__ == '__main__':
    app.run()
