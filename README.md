# FinBERT: Financial Sentiment Analysis with BERT

---
## ⚠️ **IMPORTANT LICENSING NOTICE**

**Effective Date**: 12 October 2026

### **What the Apache-2.0 License Covers**
The Apache-2.0 license in this repository applies **ONLY to the training 
code, scripts, and notebooks** contained in this repository. 

**The Apache-2.0 license does NOT cover**:
- The **pre-trained model weights** (hosted on Hugging Face)
- The **training dataset** (Financial PhraseBank)

### **Licensor**
The code in this repository is licensed by **MIH AI B.V.** under the 
Apache License, Version 2.0.

### **Model Weights**
The pre-trained FinBERT model weights are available on Hugging Face. 
Please review the **[model card on Hugging Face](https://huggingface.co/ProsusAI/finbert)** 
for licensing terms and restrictions that apply to the model weights.

### **Training Data**
This model was trained using the Financial PhraseBank dataset, which has 
its own licensing terms. Users are solely responsible for ensuring 
compliance with any third-party data licenses.

### **No Warranty**
This project is provided "AS IS" without warranty of any kind. This is 
not an official Prosus product.

---

FinBERT sentiment analysis model is now available on Hugging Face model hub. You can get the model [here](https://huggingface.co/ProsusAI/finbert). 

FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training
 the [BERT](https://arxiv.org/pdf/1810.04805.pdf) language model in the finance domain, using a large financial corpus and thereby fine-tuning
  it for financial sentiment classification. For the details, please see 
  [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/pdf/1908.10063.pdf).

## Installing
 Install the dependencies by creating the Conda environment `finbert` from the given `environment.yml` file and
 activating it.
```bash
conda env create -f environment.yml
conda activate finbert
```

## Models

### Pre-trained Model Weights

Pre-trained FinBERT model weights are available on Hugging Face:

**🔗 [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)**

Please review the model card for licensing terms and usage restrictions 
before downloading or using the model weights.

The code in this repository loads the model directly from Hugging Face by its id:
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
```
To use a model you trained yourself, pass its local directory instead of `'ProsusAI/finbert'`.

## Datasets
There are two datasets used for FinBERT. The language model further training is done on a subset of Reuters TRC2 
dataset. This dataset is not public, but researchers can apply for access 
[here](https://trec.nist.gov/data/reuters/reuters.html).

For the sentiment analysis, we used Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).
 The dataset can be downloaded from this [link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
 If you want to train the model on the same dataset, after downloading it, you should create three files under the 
 `data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`. 
To create these files, do the following steps:
- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the [datasets script](scripts/datasets.py):
```python scripts/datasets.py --data_path <path to Sentences_50Agree.txt>```

## Training the model
Training is done in `finbert_training.ipynb` notebook. Training starts from `bert-base-uncased` by default; you can
 set `bm` in the notebook to your own further pre-trained language model instead. The trained model will
 be saved to `models/classifier_model/finbert-sentiment`. You can find the training parameters in the notebook as follows:
```python
config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4.0,
                   model_dir=cl_path,
                   max_seq_length = 64,
                   train_batch_size = 32,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True )
```
The last two parameters `discriminate` and `gradual_unfreeze` determine whether to apply the corresponding technique 
against catastrophic forgetting.

## Getting predictions
We provide a script to quickly get sentiment predictions using FinBERT. Given a .txt file, `predict.py` produces a .csv file including the sentences in the text, corresponding softmax probabilities for three labels, actual prediction and sentiment score (which is calculated with: probability of positive - probability of negative).

Here's an example with the provided example text: `test.txt`. From the command line, simply run:
```bash
python scripts/predict.py --text_path test.txt --output_dir output/
```
By default this uses the `ProsusAI/finbert` model from Hugging Face. To use your own trained model, add
`--model_path models/classifier_model/finbert-sentiment`.

## Disclaimer
This is not an official Prosus product. It is the outcome of an intern research project in Prosus AI team.
### About Prosus 
Prosus is a global consumer internet group and one of the largest technology investors in the world. Operating and
 investing globally in markets with long-term growth potential, Prosus builds leading consumer internet companies that empower people and enrich communities.
For more information, please visit [www.prosus.com](www.prosus.com).

## Contact information
Please contact Dogu Araci `dogu.araci[at]prosus[dot]com` and Zulkuf Genc `zulkuf.genc[at]prosus[dot]com` about
 any FinBERT related issues and questions.
