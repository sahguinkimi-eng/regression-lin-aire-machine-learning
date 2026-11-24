"""Régression linéaire multiple avec scikit-learn: prédicrion du prix d'une maison en
fonction de sa superficie, nombre de chambre et l'anciennité"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
#données
X=np.array([[50,2,10],[70,3,8],[90,3,5],[110,4,4],[130,5,2],[150,5,1]])
y=np.array([13,19,23,29,36,40])
#Modèle
modele=LinearRegression()
#Entrainement
modele.fit(X,y)

#Les paramètres
print("Intercept:",modele.intercept_)
print("Coéfficients:",modele.coef_)

#Prédiction
y_pred=modele.predict(X)
#Métriques
print('MSE:',mean_squared_error(y,y_pred))
print("MAS:",mean_absolute_error(y,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y,y_pred)))
print("R²:",r2_score(y,y_pred))

#Nouvelle maison 
nouvelle_maison=np.array([[100,4,3]])
print("Prix prédit :",modele.predict(nouvelle_maison)[0])