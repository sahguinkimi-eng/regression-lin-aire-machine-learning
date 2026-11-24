"""Régression linéaire multiple avec numpy"""
import numpy as np
#Les données
X=np.array([[50,2,10],[70,3,8],[90,3,5],[110,4,4],[130,5,2],[150,5,1]])
y=np.array([13,19,23,29,36,40])
#Ajout de la colonne de 1 pour l'intercept
X_b=np.c_[np.ones(X.shape[0]),X]
#Calcul de la pseudo-inverse
beta=np.linalg.pinv(X_b)@y
print("Coefficients:",beta)
#prédiction
y_pred=X_b@beta
print("Valeurs prédites :",y_pred)

#Nouvelle maison avec 100m²,4 chambres et 3ans
nouvelle_maison=np.array([1,100,4,3])
prediction=nouvelle_maison@beta
print("Prix extimé:",prediction)