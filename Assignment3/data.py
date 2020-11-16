import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv("./LBW_Dataset.csv")
df = pd.read_csv("./scem.csv")

# TODO: better pre-processing
# df = df.fillna(df.mean())
# df = df.dropna()

#Mean
# print('Mean')
# df = df.fillna(df.mean())
#Median
# print('Median')
# df= df.fillna(df.median())
#Zero
# print('Zero')
# df = df.fillna(0)

#GroupBy
grouped = df.groupby('Community')
transformed = grouped.transform(lambda x: x.fillna(x.mean()))
transformed['Community'] = df['Community']
transformed = transformed[['Community', 'Age', 'Weight', 'Delivery phase', 'HB', 'IFA', 'BP','Education',	'Residence',	'Result']]
transformed['Residence'] = df['Residence']
transformed['Education'] = df['Education']

# transformed = transformed.fillna(transformed.mode())
transformed['Residence'].fillna(transformed['Residence'].mode()[0], inplace=True)
transformed['Education'].fillna(transformed['Education'].mode()[0], inplace=True)
# transformed

df = transformed

columns = list(df.columns)
columns.remove("Result")

X = df[columns]
y = df[["Result"]]

# Transform to mean 0 and std 1 (works best for NNs)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)
