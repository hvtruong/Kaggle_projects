import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.load_data import load_data
from data.preprocess import preprocess
from features.features import build_features
from models.models import *
from models.train import train_model

RAW_DATA = "src/data/raw/train.csv"

first_chunk = True
reader = load_data(RAW_DATA, chunksize=10000)

model = xgboostModel()

for chunk in reader:
    df = preprocess(chunk)
    df_processed = build_features(df)
    
    train_model(df_processed, target_col='Survived', model=model)