### document-classification-task

In this project, I have trained a Random Forest Classifier on documents containing hashed data. I have created a Flask app
which takes in an input document and tries to classify it based on the target labels available at training time.

The `train_model.py` used the given shuffled data to train the classifier and saves the `tf-idf-vectorizer` and the `trained model` so that they can be used later for testing/classifying live samples.

If running the `app.py` file locally, it starts a flask app deployed locally at http://127.0.0.1:5000/ and this app can handle both GET queries and also POST requests. 

The model is deployed and can be tested at https://rnagula-document-classifier.herokuapp.com.
