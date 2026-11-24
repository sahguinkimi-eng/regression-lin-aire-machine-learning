"""Régrsion linéaire simple, une seule variable
y=ax+b
a=sum((xi-x_moy)(yi-y_moy))/sum((xi-x_moy)**2)) et
b=y_moy-a*x_moy après la dérivéee de 
matrice Jacobienne à l'optimum"""
#import des package
import numpy as np
X=np.array([50,70,90,110,130])
Y=np.array([12,16,20,24,28])
# Calcule de la moyenne
x_moy=np.mean(X)
y_moy=np.mean(Y)
#calcul de  pente a
a=np.sum((X-x_moy)*(Y-y_moy))/np.sum((X-x_moy)**2)
#Calcul de l'ordonnée à l'origine
b=y_moy-a*x_moy
#Modèle
print("La pente de ce modèle est égale à:",a)
print("Ordonnée à l'orignine est égale à:",b)
test=input("Voulez-vous tester le modèle?\n Si oui, taper oui,\n sinon taper non:")
if test.lower()=='oui':
    x_nouveau=float(input("Enter your value X to predict:"))
    y_nouveau=a*x_nouveau+b
    print(f"Avec une superficie de {x_nouveau} m², le montant est estimé à {y_nouveau} millions  hors TVA")
else:
    print("Merci !!! je reste à votre service pour vos prochains démande")