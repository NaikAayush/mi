import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./LBW_Dataset.csv")

# TODO: better pre-processing
df = df.dropna()

columns = list(df.columns)
columns.remove("Result")

X = df[columns]
y = df[["Result"]]

# Transform to mean 0 and std 1 (works best for NNs)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)
