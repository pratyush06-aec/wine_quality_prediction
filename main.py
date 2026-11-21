import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('wine_quality_merged.csv')

# print(dataframe.head())
# print(dataframe.shape)

dataframe= dataframe.drop(columns= ['type'])

# print(dataframe.isnull().sum().sum())
# print(dataframe.describe())
# print(dataframe.head())

# plt.figure(figsize=(10, 6))
# sns.countplot(data=dataframe, x='quality')

# plt.title('Distribution of Wine Quality')
# plt.xlabel('Wine Quality')
# plt.ylabel('Number of Samples')

# plt.show()

## We can plot the visualization of the features in the dataset to see how they relate to the quality of wine.
## The best is by plotting a heatmap of the correlation matrix of the features in the dataset. This will help us to see which features are more correlated with the quality of wine.

x= dataframe.drop(columns=['quality'])
y= dataframe['quality'].apply(lambda x: 1 if x >= 6 else 0)  # Convert quality to binary classification (1 for good quality, 0 for bad quality)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# print(x.shape, x_train.shape, x_test.shape) 
# print(y.shape, y_train.shape, y_test.shape)

model= RandomForestClassifier()

model.fit(x_train, y_train)

x_test_prediction= model.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction, y_test)
test_data_precision= precision_score(x_test_prediction, y_test)
test_data_recall= recall_score(x_test_prediction, y_test)
test_data_f1= f1_score(x_test_prediction, y_test)
test_data_confusion_matrix= confusion_matrix(x_test_prediction, y_test)
test_data_classification_report= classification_report(x_test_prediction, y_test)

print("Accuracy score of the test data is: ", test_data_accuracy)
print("Precision score of the test data is: ", test_data_precision)
print("Recall score of the test data is: ", test_data_recall)
print("F1 score of the test data is: ", test_data_f1)
print("Confusion matrix of the test data is: \n", test_data_confusion_matrix)
print("Classification report of the test data is: \n", test_data_classification_report)