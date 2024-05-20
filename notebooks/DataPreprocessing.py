import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader, TensorDataset


def Data_preprocessing(csv_path: str, seed: int, batch_size: int, kfolds: int = 5):
    # Load data
    df = pd.read_csv(csv_path)

    # Check for missing values
    if df.isnull().values.any():
        df = df.dropna()  # Drop rows with any missing values if any

    # Drop irrelevant feature
    df.drop('Time', axis=1, inplace=True)

    # Separate features and labels
    normal_data = df[df['Class'] == 0]
    fraud_data = df[df['Class'] == 1]
    X_normal = normal_data.drop('Class', axis=1)
    y_normal = normal_data['Class']
    X_fraud = fraud_data.drop('Class', axis=1)
    y_fraud = fraud_data['Class']

    input_size = X_normal.shape[1]

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    splits = []

    for train_index, val_index in kf.split(X_normal):
        X_train, X_val_normal = X_normal.iloc[train_index], X_normal.iloc[val_index]
        y_train, y_val_normal = y_normal.iloc[train_index], y_normal.iloc[val_index]

        # Combine normal and fraud data for validation
        X_val = pd.concat([X_val_normal, X_fraud], axis=0)
        y_val = pd.concat([y_val_normal, y_fraud], axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        splits.append((train_loader, val_loader, y_val.values, input_size))

        # Print details
        print(f"Fold {len(splits)}")
        print(f"  Training data size: {len(X_train)}")
        print(f"  Testing data size: {len(X_val)}")
        print(f"  Labels 0 in training data: {sum(y_train == 0)}")
        print(f"  Labels 1 in testing data: {sum(y_val == 1)}")

    return splits

# splits = Data_preprocessing('creditcard.csv', 42, 32, 5)
