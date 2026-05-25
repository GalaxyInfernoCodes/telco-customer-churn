# %% [markdown]
# # Tutorial: The Misleading Nature of Accuracy on Imbalanced Data
# In this notebook, we'll train a quick model on the Telco Customer Churn dataset without any class balancing. 
# We'll see how looking only at "accuracy" can give a false sense of performance.

# %%
import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## 1. Load Data from DuckDB
# The data has already been split and ingested into our DuckDB database. We will load the training and test sets.

# %%
# Connect to DuckDB
# Assuming we run this from the notebooks/ directory
con = duckdb.connect('../data/telco.duckdb')

# Load train and test sets
train_df = con.execute("SELECT * FROM train").df()
test_df = con.execute("SELECT * FROM test").df()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Check class distribution
print("\nClass distribution in training data:")
print(train_df['Churn'].value_counts(normalize=True) * 100)

# %% [markdown]
# As we can see, the data is imbalanced. About 73% of the customers did not churn, while only 27% did. 

# %% [markdown]
# ## 2. Data Preprocessing
# We'll do some quick preprocessing to get the data ready for a machine learning model.
# - Convert TotalCharges to numeric (some values are empty strings)
# - Drop customerID as it's not a useful predictive feature
# - Map 'Churn' to 1 (Yes) and 0 (No)
# - One-hot encode categorical features

# %%
def preprocess_data(df):
    df = df.copy()
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Convert TotalCharges to numeric, coercing errors to NaN and filling with 0
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', ''), errors='coerce').fillna(0)
        
    # Map target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)

# Separate features and target
X_train = train_processed.drop('Churn', axis=1)
y_train = train_processed['Churn']

X_test = test_processed.drop('Churn', axis=1)
y_test = test_processed['Churn']

# One-hot encode categorical variables
# Align train and test columns just in case there are missing categories in test
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='inner', axis=1)

print(f"Features after encoding: {X_train_encoded.shape[1]}")

# %% [markdown]
# ## 3. Train a Quick Model (No Balancing)
# We'll train a Random Forest classifier using the default settings, with no handling of the class imbalance.

# %%
# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

# %% [markdown]
# ## 4. Evaluate the Model
# Let's see why accuracy is misleading!

# %%
# Predictions
y_pred = model.predict(X_test_encoded)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# %% [markdown]
# **Wait, ~79% accuracy? That sounds great!**
# But let's look closer using a confusion matrix and classification report.

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn (0)', 'Churn (1)'], 
            yticklabels=['No Churn (0)', 'Churn (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# %%
# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))

# %% [markdown]
# ### The Catch
# While the overall accuracy is high (~79%), look at the **Recall for the 'Churn (1)' class**.
# The model is very good at predicting when a customer will *not* churn (the majority class), 
# but it performs poorly at identifying the customers who *will* churn (the minority class). 
# 
# If the business goal is to identify churners to offer them incentives to stay, this model 
# would miss almost half of them! This demonstrates why relying solely on accuracy for 
# imbalanced datasets is highly misleading.

# %% [markdown]
# ## 5. Visualizing Decisions with a Decision Tree
# Let's train a simple Decision Tree (max_depth=3 for readability) to see how the model makes decisions on this imbalanced data.

# %%
# Train a decision tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_encoded, y_train)

# Evaluate
dt_y_pred = dt_model.predict(X_test_encoded)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, dt_y_pred, target_names=['No Churn (0)', 'Churn (1)']))

# %%
# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=X_train_encoded.columns,
          class_names=['No Churn', 'Churn'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree (max_depth=3)')
plt.show()
