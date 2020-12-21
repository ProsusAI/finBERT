import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('--data_path', type=str, help='Path to the text file.')

args = parser.parse_args()
data = pd.read_csv(args.data_path, sep='.@', names=['text','label'])

train, test = train_test_split(data, test_size=0.2, random_state=0)
train, valid = train_test_split(train, test_size=0.1, random_state=0)

train.to_csv('data/train.csv',index=False,sep='\t')
test.to_csv('data/test.csv',index=False,sep='\t')
valid.to_csv('data/validation.csv',index=False,sep='\t')