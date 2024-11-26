
# %!pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("data.csv")
df


import re

def convert_to_snake_case(column_name):
    # Add a space before capital letters and convert to lowercase
    spaced = re.sub(r'(?<!^)(?=[A-Z])', '_', column_name).lower()
    return spaced


df.columns = [convert_to_snake_case(col) for col in df.columns]


df.head(3)


df.dtypes

# %%
binary_features = ['diabetes','blood_pressure_problems','any_transplants', 'any_chronic_diseases', 'known_allergies', 'history_of_cancer_in_family', 'number_of_major_surgeries']
numerical_features = ['age','height','weight','premium_price']

# %% [markdown]
# The dataset contains a mix of binary and numerical features. The binary features include 'diabetes', 'blood_pressure_problems', 'any_transplants', 'any_chronic_diseases', 'known_allergies', 'history_of_cancer_in_family', and 'number_of_major_surgeries', which indicate the presence (1) or absence (0) of these conditions or events. The numerical features include 'age', 'height', 'weight', and 'premium_price', which are represented as integers. These numerical features provide quantitative data about the individuals, such as their age in years, height in centimeters, weight in kilograms, and the premium price of their medical insurance. This combination of binary and numerical features allows for a comprehensive analysis of the factors that may influence medical insurance prices.

# %%
#check for missing values
df.isnull().sum()

# %% [markdown]
# #### Exploratory data analysis (EDA)

# %%
df[numerical_features].describe().round(2).transpose()

# %%
df["premium_price"].value_counts()

# %% [markdown]
# A correlation matrix is used to visualize the relationships between different features.

# %%
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True,linewidths=.5, cmap="Reds")

# %% [markdown]
# A scatter plot is used to show the relationship between height and weight.

# %%
def plot_scatter_for_numerical_variables(df, numerical_variables):
    for var in numerical_variables:
        if var != "premium_price":
            plt.figure(figsize=(10, 6))
            scatter = sns.scatterplot(data=df, x=var, y="premium_price", s=100, palette="deep")
            plt.title(f"{var.capitalize()} vs Premium Price", fontsize=16)
            plt.xlabel(f"{var.capitalize()}", fontsize=14)
            plt.ylabel("Premium Price", fontsize=14)
            plt.grid(True)
            plt.show()

plot_scatter_for_numerical_variables(df, numerical_features)


sns.displot(x='age',data=df,aspect=10/7,kde=True)

# %% [markdown]
# Distribution of price

# %%
sns.displot(x='premium_price',data=df,aspect=10/7,kde=True)

# %% [markdown]
# Feature Engineering

# %%
#Creating salary-bins to visualize distribution of Premium Price and Age

pr_lab=['low','average','high']
df['premium_label']=pr_bins=pd.cut(df['premium_price'],bins=3,labels=pr_lab,precision=0)
df['age_label']=pr_bins=pd.cut(df['age'],bins=3,labels=pr_lab,precision=0)
df['weight_label']=pr_bins=pd.cut(df['weight'],bins=3,labels=pr_lab,precision=0)
df['height_label']=pr_bins=pd.cut(df['height'],bins=3,labels=pr_lab,precision=0)

# print("Premium Price Bins:", pd.cut(df['premium_price'], bins=3).unique().sort_values())
# print("Age Bins:", pd.cut(df['age'], bins=3).unique().sort_values())
# print("Weight Bins:", pd.cut(df['weight'], bins=3).unique().sort_values())
# print("Height Bins:", pd.cut(df['height'], bins=3).unique().sort_values())

# %%
df.columns

# %%
df.head(3)

# %% [markdown]
# Number of people in each premium-label based on their age-group

# %%
fig,ax=plt.subplots(figsize=(12,6))
sns.countplot(x='premium_label',hue='age_label',data=df,ax=ax)

# %% [markdown]
# Avg. price paid by people in each age category for their health insurance

# %%
df.groupby(['age_label'])['premium_price'].mean().plot(kind='bar')

# %% [markdown]
# Converting new categorical columns to numeric ones

# %%
df.drop(['premium_label','age','height','weight'],axis=1,inplace=True)
df = pd.get_dummies(data=df,columns=['age_label','weight_label', 'height_label'])

# %%
df.columns

# %%
df.dtypes

# %%
# List of columns to be one-hot encoded
columns_to_encode = ['diabetes', 'blood_pressure_problems', 'any_transplants', 'any_chronic_diseases', 'known_allergies', 'history_of_cancer_in_family', 'number_of_major_surgeries']

# Perform one-hot encoding
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Display the first few rows of the encoded dataframe
df.head()

# %% [markdown]
# ### Feature importance (MSI)

# %% [markdown]
# This section of the code is responsible for scaling the dataset.
# Scaling is a crucial step in data preprocessing for machine learning models.
# It ensures that all features contribute equally to the model's performance by normalizing the range of the data.
# This helps in improving the convergence speed of gradient descent and the overall performance of the model.
# Common scaling techniques include Standardization (z-score normalization) and Min-Max scaling.

# %%
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
xsc=sc.fit_transform(df)
xsc=pd.DataFrame(xsc,columns=df.columns)

# %% [markdown]
# Selecting top 15 features which are aligned with premium-price

# %%
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(xsc, xsc['premium_price'])
mi_scores = pd.Series(mi_scores, name="MI Scores",index=xsc.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores.round(2)

mi_scores_filtered = mi_scores[mi_scores > 0]

# %%
features = list((mi_scores_filtered).index)
df_final=df[features]

# %% [markdown]
# ### Setting up the validation framework

# %%
from sklearn.model_selection import train_test_split

# Split the original dataframe 'df' into two parts: 
# 'df_full_train' (80% of the data) and 'df_test' (20% of the data)
df_full_train, df_test = train_test_split(df_final, test_size=0.2, random_state=1)

# Further split 'df_full_train' into 'df_train' (75% of df_full_train) 
# and 'df_val' (25% of df_full_train)
# This means 'df_train' is 60% of the original data and 'df_val' is 20% of the original data
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# %%
len(df_train), len(df_val), len(df_test)

# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.premium_price.values
y_val = df_val.premium_price.values
y_test = df_test.premium_price.values

del df_train['premium_price']   
del df_val['premium_price']
del df_test['premium_price']


# %%
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# %%
def evaluate_model_performance(regressor, model_name):
    regressor.fit(df_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = regressor.predict(df_val)
    
    return regressor

# %%
ridge= Ridge()
ridge_model = evaluate_model_performance(ridge, "Lasso Regression")

# Manually tune Ridge
# Define alpha values to test
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
best_alpha_ridge = None
best_mse_ridge = float('inf')
for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=1)
    ridge.fit(df_train, y_train)
    y_val_pred = ridge.predict(df_val)
    mse = mean_squared_error(y_val, y_val_pred).round(2)
    if mse < best_mse_ridge:
        best_mse_ridge = mse
        best_alpha_ridge = alpha

print(f"Best Alpha for Ridge: {best_alpha_ridge}, MSE: {best_mse_ridge}")

# Best Alpha for Ridge: 1, MSE: 14789872.61
# 
# ### Use the model on the test set

# %%
#Concatenate the training and validation dataframes
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = df_full_train
X_full_train.shape

# %%
y_full_train = np.concatenate([y_train, y_val]) 
y_full_train.shape

# %%
df_test.shape

# %%
y_test.shape

# %%
alpha=1
ridge = Ridge(alpha=alpha, random_state=1)
print(f"Training Ridge Regression model with alpha={alpha}...")
ridge.fit(X_full_train, y_full_train)
print("Model training completed.")

print("Predicting on the test set...")
y_test_pred = ridge.predict(df_test)
mse = mean_squared_error(y_test, y_test_pred).round(2)
print(f"Mean Squared Error on the test set: {mse}")

# %%
# Select a single row from the test set
single_row = df_test.iloc[0]

# Get the actual premium price for the selected row
actual_premium_price = y_test[0]

# Reshape the single row to match the expected input shape for the model
single_row_reshaped = single_row.values.reshape(1, -1)

# Predict the premium price using the trained Ridge model
predicted_premium_price = ridge.predict(single_row_reshaped)[0]

test_data = single_row_reshaped
# %% [markdown]
# ### Save the model

# %%
import pickle
output_file = f'ridge_model.bin'
output_file

# %%
# Open the file in write-binary mode
f_out = open(output_file, 'wb') 

# Serialize and save the DictVectorizer and model to the file
pickle.dump((ridge,test_data), f_out)

# Close the file
f_out.close()

print(f"The model is saved to {output_file}")
