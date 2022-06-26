import pandas as pd 
#%% Load the dataset

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

ID = train["Loan_ID"]
exclude = "Loan_ID"
dependent_variable = 'Loan_Status'

df = train[train.columns.difference([exclude, dependent_variable])]

def know_your_data(df, target):
    print(f"Input dataframe has {df.shape[0]} rows and {df.shape[1]} columns\n")
    print("=" * 100)    
    print("column names: \n", list(df.columns), "\n")
    print("=" * 100) 
    print("Missing values present in each column: \n", df.isnull().sum(), "\n")
    print("=" * 100) 
    if target in df.columns.to_list():
        print("Target column class distribution: \n", df[target].value_counts(normalize = True), "\n")
        print("=" * 100) 
    print("Data types in data: \n", df.dtypes, "\n")
    print("=" * 100) 
    print("Data Summary: \n", df.describe().T)
    
    
know_your_data(train, 'Loan_Status')


for i in train.columns:
    print("=" * 50)
    print(i, train[i].nunique())


categoricals = []
for col, col_type in df.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
        df[col].fillna("Missing", inplace=True)
    else:
        df[col].fillna(0, inplace=True)
        
print("cateorical column in dataset are:\n", categoricals)
print("\n\nCheck if any missing value is present after imputing values in dataset: ", df.isnull().sum().sum())


df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=False)



from sklearn.linear_model import LogisticRegression

x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = train[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)


import joblib
joblib.dump(lr, 'model.pkl')


model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')