import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree


file_path = r"C:\Users\kakil\OneDrive\Desktop\prodigy\task3\bank-additional\bank-additional-full.csv"  
df = pd.read_csv(file_path, sep=';')


print(df.head())


df_encoded = pd.get_dummies(df, drop_first=True)


X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.title("Decision Tree Classifier for Bank Marketing Dataset")
plt.show()
