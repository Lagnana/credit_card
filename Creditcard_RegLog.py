# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:41:41 2025

@author: Lagnana
"""

# I.  Chargement et aperçu des données

import pandas as pd

creditcard = pd.read_excel('/creditcard.xlsx')
print(creditcard.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# II. Analyse exploratoire 
## II.1. Informations générales 

# Dimensions du jeu de données
print(f'Dimensions: {creditcard.shape}')

# Types de données
print(creditcard.dtypes)

# Vérification des valeurs manquantes
print(f'Verification de NAN: {creditcard.isnull().sum().sum()}')

# II.2. Statistique descriptive

# Statistiques descriptives
print(creditcard.describe())

# II.3. Distribution de la variable cible 

# Distribution de la variable cible
count= creditcard['Class'].value_counts(normalize=True)

#Diagramme en barre verticale de la variabkle classe
t = pd.crosstab(creditcard.Class, 'freq')
t.plot.bar()
t.plot.pie(subplots = True, figsize=(3,3))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# III.  Visualisation des données

# III.1. Histogramme

import matplotlib.pyplot as plt

# Histogramme des variables 
creditcard.hist(figsize=(20,20))

plt.show()

# III.2.  Boxplot des variables numériques

import seaborn as sns

# Boxplots des features
plt.figure(figsize = (10,8))
plt.subplot(3,2,1)
sns.boxplot(creditcard.iloc[:, 0:5])
plt.subplot(3,2,2)
sns.boxplot(creditcard.iloc[:, 5:10])
plt.subplot(3,2,3)
sns.boxplot(creditcard.iloc[:, 10:15])
plt.subplot(3,2,4)
sns.boxplot(creditcard.iloc[:, 15:20])
plt.subplot(3,2,5)
sns.boxplot(creditcard.iloc[:, 20:25])
plt.subplot(3,2,6)
sns.boxplot(creditcard.iloc[:, 25:30])

# III.3. Matrice de Corrélation

# Matrice de corrélation
corr_matrix = creditcard.corr()
# Heatmap de la matrice de corrélation
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap = 'coolwarm', annot = False)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# IV.   Analyse multivariée

# IV.1  Sélection les meilleurs variables
from sklearn.ensemble import RandomForestClassifier

# Modèle de Forêt aléatoire 
X = creditcard.drop(columns = ['Class'])
y = creditcard['Class']

model_rfc = RandomForestClassifier()
model_rfc.fit(X,y)

# Features importantes
importances = model_rfc.feature_importances_
features_names = X.columns
feature_importances_df = pd.DataFrame({'Feature': features_names, 'Importance': importances})

# Tri des features par importance décroissante
feature_importances_df_tri = feature_importances_df.sort_values(by='Importance', ascending=False)
#print(feature_importances_df_tri)

# Visualisation
plt.figure(figsize=(10,8))
sns.barplot(x = 'Importance', y = 'Feature', data = feature_importances_df_tri)
plt.title('Importance des variables')

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# V. Modélisation prédictive
# V.1. Modèle de regression logistique

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Entraînement du modèle
model_log = LogisticRegression()
model_log.fit(X_train, y_train)

# Prédictions
y_pred = model_log.predict(X_test)
y_prob = model_log.predict_proba(X_test)[:, 1] #probabilité prédite

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# V.2. Evaluation du modèle logistique avec les métriques

# V.2.1. Matrice de confusion

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Rapport de classification
print(classification_report(y_test, y_pred))
# Confusion matrix
ConfusionMatrixDisplay.from_estimator(model_log, X_test, y_test)


# V.2.2. Evaluation avec la courbe de ROC

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# V.2.2.a. Calcul de métriques

# Calcul de la courbe ROC
FPR, TPR, thresholds1 = roc_curve(y_test, y_prob)

# Calcul de l'AUC (Area Under the Curve) de ROC
test_auc_roc = roc_auc_score(y_test, y_prob)

# V.2.2.b. Calcul de seuil optimal

# Calculer la somme de TPR et (1 - FPR) pour chaque seuil
optimal_idx1 = np.argmax(TPR - FPR)  # Trouver l'index du seuil optimal
optimal_threshold1 = thresholds1[optimal_idx1]  # Le meilleur seuil est 0.024
optimal_tpr = TPR[optimal_idx1]  # Sensibilité au meilleur seuil = 0.821
optimal_fpr = FPR[optimal_idx1]  # Spécificité au meilleur seuil = 0.021

# V.2.2.c. Visualisation de la courbe ROC

# Tracer la courbe ROC
plt.figure(figsize=(12, 6))

plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')  # Ligne pour une prédiction aléatoire
plt.plot(FPR, TPR, label='ROC curve (AUC = %0.2f)' % test_auc_roc)  # Ajouter la courbe ROC

# Ajouter un point pour le seuil optimal
plt.scatter(optimal_fpr, optimal_tpr, color='red', label=f'Optimal threshold: {optimal_threshold1:.2f}', zorder=5)

# Ajouter les labels, titre et légende
plt.xlabel('False Positive Rate (1 - Spécificité)')
plt.ylabel('True Positive Rate (Sensibilité)')
plt.title('ROC Curve ')
plt.legend()

plt.show()


# V.2.2.d. Matrice de confusion au seuil optimal de ROC curve

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Appliquer le seuil de 0,02 pour obtenir les prédictions binaires
seuil_1 = 0.02
y_pred_1 = (y_prob >= seuil_1).astype(int)

# Calculer la matrice de confusion
confus_mat = confusion_matrix(y_test, y_pred_1)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=confus_mat, display_labels=model_log.classes_)
disp.plot(cmap='viridis')  # Vous pouvez choisir une autre colormap si vous le souhaitez
plt.title(f'Matrice de confusion au seuil de {seuil_1}')

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# V.2.3. Evaluation avec la courbe precision-recall

# V.2.3.a. Calcul de métriques

# Calcul de courbe Precision-Rappel
precisions, recalls, thresholds2 = precision_recall_curve(y_test, y_prob)
# Calcul de l'aire sous la courbe precision-rappel
pr_auc = auc(recalls, precisions) 

# Calcul de la moyenne harmonique F1-score
F1_score = 2*(recalls * precisions)/(recalls + precisions)

# V.2.3.b. Calcul de seuil optimal

# Calculer la somme de precision et recall pour chaque seuil
optimal_idx2 = F1_score.argmax() # Trouver l'index du seuil optimal
optimal_threshold2 = thresholds2[optimal_idx2] # Le meilleur seuil
optimal_precision = precisions[optimal_idx2] # Sensibilité au meilleur seuil
optimal_recall = recalls[optimal_idx2] # Sensibilité au meilleur seuil

# V.2.3.c. Visualisation de la courbe Precision-Rappel

# Tracer la courbe de precision-recall
plt.figure(figsize=(8, 6))

plt.plot([0, 1], [1, 0], linestyle='--', label='No Skill') # Tracer le modele sans competence
plt.plot(recalls, precisions, marker='.', label='PR curve (AUC = %.3f)' % pr_auc) # Tracer PR-Curve de la regression logistique

# Ajouter un point optimal 
plt.scatter(optimal_recall, optimal_precision, color='red', label = f'optimal threshold:{optimal_threshold2:.2f}', zorder=5)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

plt.show()

# V.2.3.d. Matrice de confusion au seuil optimal de Precision-Recall curve

# Appliquer le seuil de 0,02 pour obtenir les prédictions binaires
seuil_2 = 0.84
y_pred_2 = (y_prob >= seuil_2).astype(int)

# Calculer la matrice de confusion
confus_matrix = confusion_matrix(y_test, y_pred_2)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=confus_matrix, display_labels=model_log.classes_)
disp.plot(cmap='viridis')  # Vous pouvez choisir une autre colormap si vous le souhaitez
plt.title(f'Matrice de confusion au seuil de {seuil_2}')

plt.show()


# V.2.3.e. Visualisation de précision et rappel en fonction des seuils 

# Precision et Recall en fonction du seuil
plt.figure(figsize=(12,8))

plt.plot(thresholds2, precisions[:-1], label='Precision', color='blue')
plt.plot(thresholds2, recalls[:-1], label='Recall', color='magenta')
plt.xlabel('Thresholds')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# VI. Prédiction binaire pour chaque seuil à partir des probabilités prédites

import numpy as np

# Créer une bibliotheque
dat = {
    'True Value': y_test,
    'Predicted Probability': y_prob
}

# Créer une Dataframe
df = pd.DataFrame(dat)
# Réordonner du plus grand au plus petit
df = df.sort_values(by='Predicted Probability', ascending = False)

# Créer une liste de seuils
thresholds_list = [6e-30 , 0.4, 0.6, 0.8, 0.84]

# Prédiction binaire pour chaque seuil (1 si probabilité >= seuil, sinon 0)
for threshold in thresholds_list:
    column_name = f'S= {threshold}'
    df[column_name] = np.where(df['Predicted Probability'] >= threshold, 1, 0)

print(df)

# Enregistrer ce DataFrame dans un fichier CSV
#df.to_csv('predictions_df.csv', index=False)
