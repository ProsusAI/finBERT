from flask import Flask
from flask_cors import CORS
import sys
import optparse
import time
from flask import request
import sys
from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import nltk
import os

nltk.download('punkt_tab')
app = Flask(__name__)
CORS(app)
start = int(round(time.time()))
model = AutoModelForSequenceClassification.from_pretrained(os.environ.get('MODEL_PATH', 'ProsusAI/finbert'), num_labels=3, cache_dir=None)

@app.route("/",methods=['POST'])
def score():
    text=request.get_json()['text']
    return(predict(text, model).to_json(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
