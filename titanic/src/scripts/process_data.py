import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.load_data import load_data
from data.preprocess import preprocess
from features.features import build_features

RAW = "src/data/raw/train.csv"
OUT = "src/data/processed/processed_train.csv"

first_chunk = True
reader = load_data(RAW, chunksize=10000)

for chunk in reader:
    df = preprocess(chunk)
    df_processed = build_features(df)
    
    df_processed.to_csv(
        OUT,
        mode='a' if not first_chunk else 'w',
        header=first_chunk,
        index=False
    )
    first_chunk=False