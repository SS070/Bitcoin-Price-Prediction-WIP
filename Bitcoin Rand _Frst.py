import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Bitcoin prices.csv')

# Create the feature matrix and target variable
X = data.drop(['Date', 'Adj Close'], axis=1) # Drop the unnecessary columns
y = data['Close'] > data['Close'].shift(1) # Create the target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Plot the feature importances
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.plot(kind='barh')
plt.show()
