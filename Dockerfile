from pytorch/pytorch:latest

RUN pip install pytorch_pretrained_bert numpy pandas nltk Flask flask-cors

COPY main.py /src/main.py
COPY finbert /src/finbert
COPY models /src/models

EXPOSE  8080
CMD ["python3", "/src/main.py"]