import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
glass_info = pd.read_csv('glass.csv')
x_train = glass_info.drop("Type", axis=1)
y_train = glass_info['Type']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
q_rep = classification_report(y_test, y_pred, zero_division = 0)
print(q_rep)
print("The Accuracy of svm is: ", accuracy_score(y_test, y_pred)*100)