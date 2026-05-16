import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str, model):
    """
    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        df['preds'] = preds
        df.to_csv('predictions.csv', index=False)
        # Log params, metrics, and model
        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
        
    return model