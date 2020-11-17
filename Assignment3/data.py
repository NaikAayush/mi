import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

df = pd.read_csv("./LBW_Dataset.csv")

# TODO: better pre-processing
# df = df.fillna(df.mean())
# df = df.dropna()

# Mean
# print('Mean')
# df = df.fillna(df.mean())
# Median
# print('Median')
# df= df.fillna(df.median())
# Zero
# print('Zero')
# df = df.fillna(0)

# GroupBy
# grouped = df.groupby("Community")
# transformed = grouped.transform(lambda x: x.fillna(x.mean()))
# transformed["Community"] = df["Community"]
# transformed = transformed[
#     [
#         "Community",
#         "Age",
#         "Weight",
#         "Delivery phase",
#         "HB",
#         "IFA",
#         "BP",
#         "Education",
#         "Residence",
#         "Result",
#     ]
# ]
# transformed["Residence"] = df["Residence"]
# transformed["Education"] = df["Education"]

# # transformed = transformed.fillna(transformed.mode())
# transformed["Residence"].fillna(transformed["Residence"].mode()[0], inplace=True)
# transformed["Education"].fillna(transformed["Education"].mode()[0], inplace=True)
# transformed


# df = transformed

# df['HB'] = df['HB'].fillna(df.groupby(['Community','Age','Weight','Delivery phase','BP'])['HB'].transform('mean'))
# df['HB'] = df['HB'].fillna(df.groupby(['Community','Age','Weight','Delivery phase'])['HB'].transform('mean'))
# df['HB'] = df['HB'].fillna(df.groupby(['Community','Age','Weight'])['HB'].transform('mean'))
# df['HB'] = df['HB'].fillna(df.groupby(['Community','Age'])['HB'].transform('mean'))
# df['HB'] = df['HB'].fillna(df.groupby('Community')['HB'].transform('mean'))
# df['BP'] = df['BP'].fillna(df.groupby(['Community','Age','Weight','Delivery phase','HB'])['BP'].transform('mean'))
# df['BP'] = df['BP'].fillna(df.groupby(['Community','Age','Weight','Delivery phase'])['BP'].transform('mean'))
# df['BP'] = df['BP'].fillna(df.groupby(['Community','Age','Weight'])['BP'].transform('mean'))
# df['BP'] = df['BP'].fillna(df.groupby(['Community','Age'])['BP'].transform('mean'))
# df['BP'] = df['BP'].fillna(df.groupby('Community')['BP'].transform('mean'))
# df['Age'] = df['Age'].fillna(df.groupby('Community')['Age'].transform('mean'))
# df['Weight'] = df['Weight'].fillna(df.groupby(['Community','Age'])['Weight'].transform('mean'))
# df['Weight'] = df['Weight'].fillna(df.groupby('Community')['Weight'].transform('mean'))

#NEW
df['HB'] = df['HB'].fillna(df.groupby(['Community','Age','BP'])['HB'].transform('mean'))
df['HB'] = df['HB'].fillna(df.groupby(['Community','Age'])['HB'].transform('mean'))
df['HB'] = df['HB'].fillna(df.groupby('Community')['HB'].transform('mean'))

df['BP'] = df['BP'].fillna(df.groupby(['Community','Age','HB'])['BP'].transform('mean'))
df['BP'] = df['BP'].fillna(df.groupby(['Community','Age'])['BP'].transform('mean'))
df['BP'] = df['BP'].fillna(df.groupby('Community')['BP'].transform('mean'))

df['Age'] = df['Age'].fillna(df.groupby('Community')['Age'].transform('mean'))

df['Weight'] = df['Weight'].fillna(df.groupby(['Community','Age'])['Weight'].transform('mean'))
df['Weight'] = df['Weight'].fillna(df.groupby('Community')['Weight'].transform('mean'))
#NEWEND
# df = df.drop(['Delivery phase','Residence','Education'], axis = 1) 

df["Delivery phase"].fillna(df["Delivery phase"].mode()[0], inplace=True)
df["Residence"].fillna(df["Residence"].mode()[0], inplace=True)
df["Education"].fillna(df["Education"].mode()[0], inplace=True)
#NEWEND
df = pd.concat([df, df.loc[df[df.Result == 0].index.repeat(2)]])

columns = list(df.columns)
columns.remove("Result")

X = df[columns]
y = df[["Result"]]

# Transform to mean 0 and std 1 (works best for NNs)
# scaler = Normalizer(norm="l2")
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
