import pandas as pd
df = pd.read_excel("2021-2023 FNDDS At A Glance - Foods and Beverages.xlsx.xlsx")
    
from transformers import pipeline

# use BERT NER
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
entities = ner_pipeline("Text from a cell in the table")

from pykeen.pipeline import pipeline

# ʹ��TransEģ��ѵ��Ƕ��
result = pipeline(
    model='TransE',
    dataset='WN18RR',  # �ɻ����Լ������ݼ�
    training_loop='lcwa',
)
