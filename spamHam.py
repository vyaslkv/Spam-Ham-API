import pandas as pd
from flask import Flask, request, render_template
import requests
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
 	return render_template('home.html')
@app.route('/process',methods=['GET', 'POST'])
def nextFn():
    # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])
    df['label'] = df.label.map({'ham':0, 'spam':1})
    count_vector = CountVectorizer()

    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

    count_vector = CountVectorizer()

    # Fit the training data and then return the matrix
    training_data = count_vector.fit_transform(X_train)
    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    user_text= request.form.get('raw')
    t_data= count_vector.transform([str(user_text)])
    p = naive_bayes.predict(t_data)
    if p[0]==0:
        result="Not Spam"
        return render_template('response.html', result=result)
    else:
        result="Spam"
        return render_template('response.html', result=result)

if __name__=='__main__':
    app.run(port=50002, debug=True)
