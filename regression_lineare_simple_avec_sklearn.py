"""La régression linéaire avec scikit-learn"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

#Variable explicatives
X=np.array([[50],[70],[90],[110],[130]])
#Variable cible
Y=np.array([12,16,20,24,28])
#création du modèle 
model=LinearRegression()
#Entrainement du modèle
model.fit(X,Y)
#paramètre appris
print("Pente:",model.coef_[0])
print("Intercept:",model.intercept_)

#Prédiction
y_pred=model.predict(X)
#Métriques
print('MSE:',mean_squared_error(Y,y_pred))
print('RMSE :',np.sqrt(mean_squared_error(Y,y_pred)))
print('MAE :',mean_absolute_error(Y,y_pred))
print('R²:',r2_score(Y,y_pred))

#Visualisation
plt.scatter(X,Y,label="Données réélles")
plt.plot(X,y_pred,label='Droite de régression')
plt.xlabel("Superficie (m²)")
plt.ylabel("Prix (millions de FCFA)")
plt.title('Régrssion linéaire simple')
plt.legend()
plt.grid()
plt.show()
#Prédiction
test=input("Voulez-vous tester le modèle?\n Si oui, taper oui,\n sinon taper non:")
if test.lower()=='oui':
    x_nouveau=float(input("Enter your value X to predict:"))
    y_nouveau=model.predict([[x_nouveau]])
    print(f"Avec une superficie de {x_nouveau} m², le montant est estimé à {y_nouveau[0]} millions  hors TVA")
else:
    print("Merci !!! je reste à votre service pour vos prochains démande")