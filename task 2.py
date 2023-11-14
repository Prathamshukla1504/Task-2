import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the credit card transaction data
data = pd.read_csv("credit_card_transactions.csv")

# Separate the features and target variable
features = data.drop("is_fraud", axis=1)
target = data["is_fraud"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate the model on the testing set
predictions = classifier.predict(X_test)
confusion_matrix_results = confusion_matrix(y_test, predictions)
classification_report_results = classification_report(y_test, predictions)

print("Confusion Matrix:")
print(confusion_matrix_results)

print("\nClassification Report:")
print(classification_report_results)
