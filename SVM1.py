import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart1.csv')

# Preprocess the data
X = data.drop('target', axis=1)
y = data['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {'C': [0.01, 0.1, 1], 'gamma': [0.01, 0.1], 'kernel': ['rbf', 'linear']}

# Perform grid search
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the corresponding accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Train the SVM model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Print the evaluation of training data
print("The evaluation of training data\n")
y_train_pred = best_model.predict(X_train)
print('Training Accuracy:', accuracy_score(y_train, y_train_pred))
print('Training Precision:', precision_score(y_train, y_train_pred))
print('Training Recall:', recall_score(y_train, y_train_pred))
print('Training F1-score:', f1_score(y_train, y_train_pred))

# Evaluate the performance of the trained model on the testing data
print("The evaluation on testing data\n")
print('Testing Accuracy:', accuracy_score(y_test, y_pred))
print('Testing Precision:', precision_score(y_test, y_pred))
print('Testing Recall:', recall_score(y_test, y_pred))
print('Testing F1-score:', f1_score(y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model to disk
joblib.dump(best_model, 'heart_disease_model_svm.joblib')
joblib.dump(scaler,'scaler.joblib')
# Use the model to make predictions on new data
new_data = np.array([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]])
#prediction = best_model.predict(scaler.transform(new_data))
#print(new_data)
#if prediction[0] == 1:
    #print("Heart disease risk predicted")
#else:
    #print("No heart disease risk predicted")

# Use the model to get confidence scores (decision function) on new data
confidence_scores = best_model.decision_function(scaler.transform(new_data))

# Convert confidence scores into probabilities using Platt scaling
sigmoid = lambda x: 1 / (1 + np.exp(-x))
probabilities = sigmoid(confidence_scores)

# Percentage risk of heart disease
percentage_risk = probabilities[0] * 100

print("Percentage Risk of Heart Disease:", percentage_risk)

# Predict heart disease risk based on probability threshold
threshold = 0.5
prediction = 1 if probabilities[0] >= threshold else 0

if prediction == 1:
    print("Heart disease risk predicted")
else:
    print("No heart disease risk predicted")


# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Create age groups
age_bins = [18, 30, 40, 50, 60, 70, 80, 90]
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)

# Bar plot for heart disease distribution by age group
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', hue='target', data=data, order=age_labels)
plt.title('Heart Disease Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No Disease', 'Disease'])
plt.show()