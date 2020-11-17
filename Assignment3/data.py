import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./LBW_Dataset.csv")


# NEW
df["HB"] = df["HB"].fillna(
    df.groupby(["Community", "Age", "BP"])["HB"].transform("mean")
)
df["HB"] = df["HB"].fillna(df.groupby(["Community", "Age"])["HB"].transform("mean"))
df["HB"] = df["HB"].fillna(df.groupby("Community")["HB"].transform("mean"))

df["BP"] = df["BP"].fillna(
    df.groupby(["Community", "Age", "HB"])["BP"].transform("mean")
)
df["BP"] = df["BP"].fillna(df.groupby(["Community", "Age"])["BP"].transform("mean"))
df["BP"] = df["BP"].fillna(df.groupby("Community")["BP"].transform("mean"))

df["Age"] = df["Age"].fillna(df.groupby("Community")["Age"].transform("mean"))

df["Weight"] = df["Weight"].fillna(
    df.groupby(["Community", "Age"])["Weight"].transform("mean")
)
df["Weight"] = df["Weight"].fillna(df.groupby("Community")["Weight"].transform("mean"))
# NEWEND

df["Delivery phase"].fillna(df["Delivery phase"].mode()[0], inplace=True)
df["Residence"].fillna(df["Residence"].mode()[0], inplace=True)
df["Education"].fillna(df["Education"].mode()[0], inplace=True)
# NEWEND
df = pd.concat([df, df.loc[df[df.Result == 0].index.repeat(2)]])

columns = list(df.columns)
columns.remove("Result")

X = df[columns].to_numpy()
y = df[["Result"]].to_numpy()

# Transform to mean 0 and std 1 (works best for NNs)
scaler = StandardScaler()
X = scaler.fit_transform(X)

data = np.concatenate((X, y), axis=1)
processed_df = pd.DataFrame(data, columns=df.columns)
processed_df.to_csv("./processed.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
