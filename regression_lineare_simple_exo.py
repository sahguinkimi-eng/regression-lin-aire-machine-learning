import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import sympy as sym

# Les données
X=np.array([1,2,3,4,5,6,7]).reshape(-1,1)
Y=np.array([80,100,130,150,180,210,230])
#Entrainemement du modèle
modele=LinearRegression()
modele.fit(X,Y)

#Prédiction
y_pred=modele.predict(X)
#Métriques
print("MSE:",mean_squared_error(Y,y_pred))
print("MAE:",mean_absolute_error(Y,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(Y,y_pred)))
print("R²:",r2_score(Y,y_pred))

# equation de la droite du régression linéaire
x=sym.symbols("x")
y=modele.coef_[0]*x+modele.intercept_
print("y=",sep='')
sym.pprint(y)

#Pour 8 ans d'expériences on
y=modele.predict([[8]])
print("Avec 8 ans d'expériences on a :",y[0] ,'millions FCFA')
# Visualistion
plt.scatter(X,Y,label='Données réelles')
plt.plot(X,y_pred,label="La droite de régression linéaires simple")
plt.xlabel("Nombre d'années d'expérineces")
plt.ylabel("Le salaires en fonction du nombre d'années d'expérience")
plt.title("Régression linéaire simple")
plt.grid()
plt.legend()
plt.show()