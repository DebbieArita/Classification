# Classification
import pandas
df = pandas.read_csv('london_merged.csv')
print(df)
print(df.shape)
print(df.describe())

subset = df[['hum', 'cnt', 'is_holiday']]
array = subset.values
X = array[:, 0:2] # means all rows from columns 0...1
y = array[:, 2] # 2 is counted here

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
object = DecisionTreeClassifier()
# object = KNeighborsClassifier()
print('Model Training for Data... please wait')
object.fit(X_train, Y_train) # model is now fitted with data for training
print('Learning Completed')


# ask model to predit the
predictions = object.predict(X_test)
print(predictions)

# compare predictions with Y_test
from sklearn.metrics import  accuracy_score
print(accuracy_score(Y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, predictions))

# model failed in yes
# need improvement in yes outcome...
new = [[88, 200]]
observation = object.predict(new)
print('Predicted: ', observation)