import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

import chardet

with open("data/HCRDatabaseAnalysisStream.csv", 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

df = pd.read_csv("data/HCRDatabaseAnalysisStream.csv", encoding=encoding)
engine = create_engine("sqlite:///hcr.db")
df.to_sql("HCR", engine, index=False)