import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if not os.path.exists('data/sentiment_data'):
    os.makedirs('data/sentiment_data')

parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--data_path', type=str, help='Path to the text file.')

args = parser.parse_args()
data = pd.read_csv(args.data_path, sep='.@', names=['text','label'])

train, test = train_test_split(data, test_size=0.2, random_state=0)
train, valid = train_test_split(train, test_size=0.1, random_state=0)

train.to_csv('data/sentiment_data/train.csv',sep='\t')
test.to_csv('data/sentiment_data/test.csv',sep='\t')
valid.to_csv('data/sentiment_data/validation.csv',sep='\t')