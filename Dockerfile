from pytorch/pytorch:latest

RUN pip install numpy pandas nltk Flask flask-cors transformers

COPY main.py /src/main.py
COPY finbert /src/finbert

EXPOSE  8080
CMD ["python3", "/src/main.py"]
