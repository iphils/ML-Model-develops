import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import joblib
import pickle

# Step 1: Load data from Excel file
data = pd.read_csv('input_data_weather.csv')

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(data.columns[-1], axis=1), data[data.columns[-1]], test_size=0.2, random_state=42)

# Step 3: Define dictionary of algorithms
algorithms = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest Regresspr': RandomForestRegressor(),
    'Random Forest Classifier': RandomForestClassifier()
}

# Step 4: Prompt user to choose algorithm
print('Choose an algorithm:')
for i, name in enumerate(algorithms.keys()):
    print(f'{i+1}. {name}')
choice = int(input())

# Step 5: Train selected algorithm
selected_algorithm = list(algorithms.values())[choice-1]
selected_algorithm.fit(X_train, y_train)

# Step 6: Test model on testing data
predictions = selected_algorithm.predict(X_test)

# Step 7: Display accuracy score
print(f'Accuracy: {accuracy_score(y_test, predictions)}')

# Step 8: Save trained model to file
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(selected_algorithm, f)


# loading of the ML Model
model = joblib.load('trained_model.pkl')
feature_names=model.feature_names_in_

# create an empty DataFrame to store input data
input_data = pd.DataFrame(columns=feature_names)
# Loop over the feature names and prompt the user for input
for feature in feature_names:
    value = input(f"Enter {feature}: ")
    #input_data.append(float(value))
    input_data.loc[0, feature] = value

# use the trained model to make predictions
prediction = model.predict(input_data)

# print the predicted class
print(data.columns[-1], prediction[0])