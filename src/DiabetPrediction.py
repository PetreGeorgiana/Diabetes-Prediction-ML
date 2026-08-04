#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:04:02 2023

@author: georgianapetre
"""

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import statistics
from scipy.stats import kstest

import os
print("Folderul curent este:", os.getcwd())

dataset = pd.read_csv("Diabet_prediction.csv");
print(dataset.head())
print(dataset.info())
print(dataset.describe())


c = dataset.isnull().sum().sum()
print("Numarul de valori nule este: ",c)
print()
dupl = dataset.duplicated().sum()
print ("Numarul de dubluri este: ",dupl)
print()

dataset.drop_duplicates(inplace = True)
print(dataset.shape) 


plt.figure(figsize=(10,6))
plt.pie(dataset['Diabetes_binary'].value_counts(), labels=['Nondiabetici', 'Diabetici'], autopct='%1.2f%%', colors=['#86bf91', 'orange'])
plt.title('Procent Diabetes_binary')
plt.show()

datasetCorr=dataset.corr()
datasetCorr['Diabetes_binary'].sort_values()
dataset.drop('Diabetes_binary', axis=1).corrwith(dataset.Diabetes_binary).sort_values().plot(kind='bar', grid=True, figsize=(20, 8)
, title="Corelatia dintre variabile si Diabetes_binary",color="red")

#Decision Tree:

X = dataset.drop(labels=["GenHlth", "HighBP", "HighChol",], axis = 1).values
y = dataset['Diabetes_binary'].values



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA()
X = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print("Varianta = ", explained_variance)

train = train_test_split(dataset, test_size=0.2, random_state=0)



#Random forest:

pca2 = PCA(n_components=2)
X_2 = pca2.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2, random_state=5)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Acuratete pentru 2 componente principale: ' , accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print()

confusion = confusion_matrix(y_test, y_pred)
print("Matricea de confuzie pentru 2 componente:")
print(confusion)

pca3 = PCA(n_components=3)
X_3 = pca3.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_3, y, test_size=0.2, random_state=5)
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Acuratete pentru 3 componente principale: ' , accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print()

confusion = confusion_matrix(y_test, y_pred)
print("Matricea de confuzie pentru 3 componente:")
print(confusion)

pca4 = PCA(n_components=6)
X_4 = pca4.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_4, y, test_size=0.1, random_state=5)
classifier4 = RandomForestClassifier(max_depth=5, random_state=0)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred)
print('Acuratete pentru 6 componente principale: ' , accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print()
confusion = confusion_matrix(y_test, y_pred)
print("Matricea de confuzie pentru 6 componente:")
print(confusion)


#Decision Tree:

pca_tree = PCA(n_components = 6)
X_tree = pca_tree.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tree, y, test_size=0.1, random_state=5)
clasificator_tree = DecisionTreeClassifier(max_depth =5,  random_state=42)
clasificator_tree.fit(X_train, y_train)
ytree1_pred = clasificator_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, ytree1_pred)
print('Acuratete pentru 6 componente principale-Decision Tree: ' , accuracy_score(y_test, ytree1_pred))
print(classification_report(y_test, ytree1_pred))
print()
confusion = confusion_matrix(y_test, ytree1_pred)
print("Matricea de confuzie pentru 6 componente-Decision Tree:")
print(confusion)


    
X = dataset.drop('Diabetes_binary', axis = 1)
y = dataset['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model1 = LogisticRegression(max_iter = 1000, random_state = 42)

model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Acuratete Regresie Logistica:", accuracy)
print("Matricea de confuzie:\n", conf_matrix)
print("Raportul de clasificare:\n", class_report)

import xgboost as xgb

model = xgb.XGBClassifier(random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Acuratete XGBoost:", accuracy)
print("Matricea de confuzie:\n", conf_matrix)
print("Raportul de clasificare:\n", class_report)

from sklearn.ensemble import VotingClassifier

X = dataset.drop(labels=["GenHlth", "HighBP", "HighChol",], axis = 1).values
y = dataset['Diabetes_binary'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

voting_classifier = VotingClassifier(estimators=[
    ('Decision Tree', clasificator_tree),
    ('Random Forest', classifier4),
    ('Regresia Logistica', model1)
], voting='hard')  

voting_classifier.fit(X_train, y_train)

y_pred = voting_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratete pentru Voting Classifier incluzand Decision Tree, Random Forest si Regresia Logistica: {accuracy:.2f}")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Acuratete Voting Classifier:", accuracy)
print("Matricea de confuzie:\n", conf_matrix)
print("Raportul de clasificare:\n", class_report)

plt.figure(figsize=(8, 5))
plt.bar(['Decision Tree', 'Random Forest'], [accuracy_tree, accuracy_forest], color=['orange', '#86bf91'])
plt.xlabel('Algoritm')
plt.ylabel('Acuratete')
plt.title('Comparatie: Decision Tree vs. Random Forest')
plt.ylim(0, 1) 
plt.show()

plt.figure(figsize=(22,22))
columns = datasetCorr.nlargest(22, 'Diabetes_binary')['Diabetes_binary'].index
correlation_matrix = np.corrcoef(dataset[columns].values.T)
sns.set(font_scale=1.25)
heat_map = sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=columns.values, xticklabels=columns.values)
plt.show()

#descriu fiecare atribut printr-o histograma

ax = dataset.hist( column='Diabetes_binary', bins=[0, 0.5, 1], grid=True, 
figsize=(16,10), color='#86bf91', label=' 0 - diabetic \n 1 - nondiabetic', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Diagnostic')
plt.ylabel('Pacienti')
plt.legend()

ax = dataset.hist(column='HighBP', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Tensiune mica/mare')
plt.ylabel('Pacienti')

ax = dataset.hist(column='HighChol', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Nivel de colesterol mic/mare')
plt.ylabel('Pacienti')

ax = dataset.hist(column='CholCheck', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Verificare nivel colesterol')
plt.ylabel('Pacienti')

ax = dataset.hist(column='BMI', bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65], 
grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xlabel('Indicele de masa corporala')
plt.ylabel('Pacienti')

mean1 = dataset["BMI"].mean()
print()
print("Media coloanei BMI este:")
print(mean1)
print()
median1 = dataset["BMI"].median()
print("Mediana coloanei BMI este:")
print(median1)
print()
stdev1 = statistics.stdev(dataset["BMI"])
print("Deviatia standard a coloanei BMI este:")
print(stdev1)
print()
significance_level = 0.05
k=kstest(dataset['BMI'],'norm')
print(k)
print()

ax = dataset.hist(column='Smoker', bins=[0, 0.5, 1], grid=True, 
figsize=(16,10), bottom=True, color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('100 pachete nefumate/fumate')
plt.ylabel('Pacienti')

ax = dataset.hist(column='Stroke', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Aparitie atac vascular cerebral')
plt.ylabel('Pacienti')

ax = dataset.hist(column='HeartDiseaseorAttack', bins=[0, 0.5, 1], grid=True, 
figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Aparitie boli cardiovasculare')
plt.ylabel('Pacienti')


ax = dataset.hist(column='PhysActivity', bins=[0, 0.5, 1], grid=True, 
figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Activitate fizica')
plt.ylabel('Pacienti')

ax = dataset.hist(column='Fruits', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Consumul de fructe')
plt.ylabel('Pacienti')

ax = dataset.hist(column='Veggies', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Consumul de legume')
plt.ylabel('Pacienti')

ax = dataset.hist(column='HvyAlcoholConsump', bins=[0, 0.5, 1], grid=True, 
figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Consumul de alcool')
plt.ylabel('Pacienti')

ax = dataset.hist(column='AnyHealthcare', bins=[0, 0.5, 1], grid=True, 
figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Asigurat medical')
plt.ylabel('Pacienti')

ax = dataset.hist(column='NoDocbcCost', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Fara medic-Costuri ridicate')
plt.ylabel('Pacienti')

ax = dataset.hist(column='GenHlth', bins=5, grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('Stare generala de sanatate/ 1 - excelenta, 2 - foarte buna, 3 - buna, 4 - potrivita, 5 - rea')
plt.ylabel('Pacienti')

median2=dataset["GenHlth"].median()
print("Mediana coloanei GenHlth este: ",median2)
print()


ax = dataset.hist(column='MentHlth', bins=6, grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xlabel('Stare mintala de sanatate')
plt.ylabel('Pacienti')

ax = dataset.hist(column='PhysHlth', bins=6, grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xlabel('Stare fizica de sanatate')
plt.ylabel('Pacienti')

ax = dataset.hist(column='DiffWalk', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Dificultati de deplasare')
plt.ylabel('Pacienti')

ax = dataset.hist(column='Sex', bins=[0, 0.5, 1], grid=True, figsize=(16,10), color='#86bf91', 
label=' 0 - feminin \n 1 - masculin', zorder=1.5, rwidth=1.5)
plt.xticks([0, 1])
plt.xlabel('Genul pacientilor')
plt.ylabel('Pacienti')
plt.legend()

ax = dataset.hist(column='Education', bins=6, grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xlabel('Gradul de educatie')
plt.ylabel('Pacienti')

ax = dataset.hist(column='Age', bins=13, grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xticks([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
plt.xlabel('Varsta')
plt.ylabel('Pacienti')

mean3 = dataset["Age"].mean()
print()
print("Media coloanei Age este:")
print(mean3)
print()
median3 = dataset["Age"].median()
print("Mediana coloanei Age este:")
print(median3)
print()
stdev2 = statistics.stdev(dataset["Age"])
print("Deviatia standard a coloanei Age este:")
print(stdev2)
print()
k1=kstest(dataset['Age'],'norm')
print(k1)
print()

ax = dataset.hist(column='Income', bins=8, grid=True, figsize=(16,10), color='#86bf91', zorder=2, rwidth=2)
plt.xlabel('Venituri')
plt.ylabel('Pacienti')
print()
k2=kstest(dataset['Income'],'norm')
print(k2)

