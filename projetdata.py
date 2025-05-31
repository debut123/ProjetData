# -*- coding: utf-8 -*-
"""
Created on Tue May 27 01:03:33 2025

Nom :du chalard

Prénom : François

Projet :data




@author: frdch
"""


# importation

import math as mt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools
import statsmodels.api as sm





# initialization des variables listes de données:
    
    
Ville=[]
Temperature_minimale=[]
Temperature_maximale=[]
Hauteur_precipitations=[]
Duree_ensoleillement=[]
erreur=0

def traitement_de_donnee():
    nbdonneesincompletes=0
    
    with open("data1.csv", "r",encoding= "utf-8") as f:
        next(f)
        
        
        for ligne in f:
            meteo=ligne.strip().split(",")
            #print(meteo)
            
            if len(meteo)==5:
            
                Ville.append(meteo[0])
                Temperature_minimale.append(float(meteo[1]))
                Temperature_maximale.append(float(meteo[2]))
                Hauteur_precipitations.append(float(meteo[3]))
                Duree_ensoleillement.append(float(meteo[4]))
            else:
                nbdonneesincompletes+=1
                
    #print(Temperature_maximale)         
    return nbdonneesincompletes
                
        #print(Ville)
        
erreur=traitement_de_donnee()






Nbdonnees=len(Ville)
moyenne_Temperature_minimale=sum(Temperature_minimale)/len(Temperature_minimale)
moyenne_Temperature_maximale=sum(Temperature_maximale)/len(Temperature_maximale)
moyenne_Hauteur_precipitations=sum(Hauteur_precipitations)/len(Hauteur_precipitations)
moyenne_Duree_ensoleillement=sum(Duree_ensoleillement)/len(Duree_ensoleillement)


#=========================================================================================

# Question 1


          


def question1():
    print("Il y a exactement ", len(Ville), " de villes dans data1.csv")
    print("Il y a tant ",erreur," de données dans data1.csv qui sont incomplètes.")
    

#question1()


#=========================================================================================

# Question 2 :
    
    
def question2():
    
    print(" La ville associée à la valeur minimale de température minimale est ",end="")
    print(Ville[Temperature_minimale.index(min(Temperature_minimale))], end="")
    print(" qui a pour valeur",min(Temperature_minimale))
    
    print(" La ville associée à la valeur maximale de température minimale est ",end="")
    print(Ville[Temperature_minimale.index(max(Temperature_minimale))], end="")
    print(" qui a pour valeur",max(Temperature_minimale))
    
    print(" La ville associée à la valeur minimale de température maximale est ",end="")
    print(Ville[Temperature_maximale.index(min(Temperature_maximale))], end="")
    print(" qui a pour valeur",min(Temperature_maximale))
    
    print(" La ville associée à la valeur maximale de température maximale est ",end="")
    print(Ville[Temperature_maximale.index(max(Temperature_maximale))], end="")
    print(" qui a pour valeur",max(Temperature_maximale))
    
    print(" La ville associée à la valeur minimale de Hauteur_precipitations est ",end="")
    print(Ville[Hauteur_precipitations.index(min(Hauteur_precipitations))], end="")
    print(" qui a pour valeur",min(Hauteur_precipitations))
    
    print(" La ville associée à la valeur maximale de Hauteur_precipitations est ",end="")
    print(Ville[Hauteur_precipitations.index(max(Hauteur_precipitations))], end="")
    print(" qui a pour valeur",max(Hauteur_precipitations)) 
    
    print(" La ville associée à la valeur minimale de durée d’ensoleillement est ",end="")
    print(Ville[Duree_ensoleillement.index(min(Duree_ensoleillement))], end="")
    print(" qui a pour valeur",min(Duree_ensoleillement))
    
    print(" La ville associée à la valeur maximale de durée d’ensoleillement est ",end="")
    print(Ville[Duree_ensoleillement.index(max(Duree_ensoleillement))], end="")
    print(" qui a pour valeur",max(Duree_ensoleillement)) 
    
    
question2()

# On remarque que la valeur maximale de la température minimale est inférieur à la
# valeure minimale de la température maximale de plus on constate une grande différence de 
#durée d'ensoleillement qui peux varié de 1363 à 2854.5.

#=========================================================================================

# Question 3 :
    
    
def question3_traitement():
    somme_carres_Temperature_minimale = sum(x**2 for x in Temperature_minimale) / Nbdonnees
    somme_carres_Temperature_maximale = sum(x**2 for x in Temperature_maximale) / Nbdonnees
    somme_carres_Hauteur_precipitations = sum(x**2 for x in Hauteur_precipitations) / Nbdonnees
    somme_carres_Duree_ensoleillement = sum(x**2 for x in Duree_ensoleillement) / Nbdonnees
    
    variance_Temperature_minimale = somme_carres_Temperature_minimale - moyenne_Temperature_minimale**2
    variance_Temperature_maximale = somme_carres_Temperature_maximale - moyenne_Temperature_maximale**2
    variance_Hauteur_precipitations = somme_carres_Hauteur_precipitations - moyenne_Hauteur_precipitations**2
    variance_Duree_ensoleillement = somme_carres_Duree_ensoleillement - moyenne_Duree_ensoleillement**2
    
    return (variance_Temperature_minimale, variance_Temperature_maximale, variance_Hauteur_precipitations, variance_Duree_ensoleillement)

# Ici, on récupère les résultats et on les stocke dans des variables globales
variance_Temperature_minimale, variance_Temperature_maximale, variance_Hauteur_precipitations, variance_Duree_ensoleillement = question3_traitement()


def question3():
    
    print("la variance de Temperature_minimale est : ",variance_Temperature_minimale)
    print("la variance de Temperature_maximale est : ",variance_Temperature_maximale)
    print("la variance de Hauteur_precipitations est : ",variance_Hauteur_precipitations)
    print("la variance de Duree_ensoleillement est : ",variance_Duree_ensoleillement)
    
    
    
    
#question3()
    
# On remarque que la duréee d'ensoleillement possède une grande variance de 190704 
# ce qui signifie une grande répartion des valeurs tandis que pour la variance des
# Temperature_minimale est beaucoup plus faible de 3.123
# ce qui signifie que les valeurs sont rapprochées.

#=========================================================================================

# Question 4 :
    

def mediane():
    # Tri des listes (on fait une copie pour ne pas modifier les originaux)
    Temperature_minimale_triee = sorted(Temperature_minimale)
    Temperature_maximale_triee = sorted(Temperature_maximale)
    Hauteur_precipitations_triee = sorted(Hauteur_precipitations)
    Duree_ensoleillement_triee = sorted(Duree_ensoleillement)
    
      

    if Nbdonnees % 2 == 0:
        i1 = Nbdonnees // 2 - 1
        i2 = Nbdonnees // 2
        mediane_Temperature_minimale = (Temperature_minimale_triee[i1] + Temperature_minimale_triee[i2]) / 2
        mediane_Temperature_maximale = (Temperature_maximale_triee[i1] + Temperature_maximale_triee[i2]) / 2
        mediane_Hauteur_precipitations = (Hauteur_precipitations_triee[i1] + Hauteur_precipitations_triee[i2]) / 2
        mediane_Duree_ensoleillement = (Duree_ensoleillement_triee[i1] + Duree_ensoleillement_triee[i2]) / 2
    else:
        i = Nbdonnees // 2
        mediane_Temperature_minimale = Temperature_minimale_triee[i]
        mediane_Temperature_maximale = Temperature_maximale_triee[i]
        mediane_Hauteur_precipitations = Hauteur_precipitations_triee[i]
        mediane_Duree_ensoleillement = Duree_ensoleillement_triee[i]
        
    return (
        mediane_Temperature_minimale,
        mediane_Temperature_maximale,
        mediane_Hauteur_precipitations,
        mediane_Duree_ensoleillement
    )

(
    mediane_Temperature_minimale,
    mediane_Temperature_maximale,
    mediane_Hauteur_precipitations,
    mediane_Duree_ensoleillement
) = mediane()

    # Affichage des résultats
    
def question4():
    
    
    print("La moyenne des températures minimales est : ", moyenne_Temperature_minimale)
    print("La médiane des températures minimales est : ", mediane_Temperature_minimale)
    print("La Variance des températures minimales est : ",variance_Temperature_minimale)
    print("L'écart type des températures minimales est : ",mt.sqrt(variance_Temperature_minimale))
    
    # Affichage de l'histogramme des températures minimales
    plt.hist(Temperature_minimale, bins=10, color='skyblue', edgecolor='black')
    plt.title("Histogramme des températures minimales")
    plt.xlabel("Température minimale (°C)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()
    

#question4()


#=========================================================================================

# Question 5 :


def question5():
    
    
    print("La moyenne des Duree_ensoleillement est : ", moyenne_Duree_ensoleillement)
    print("La médiane des Duree_ensoleillement est : ", mediane_Duree_ensoleillement)
    print("La Variance des Duree_ensoleillement est : ",variance_Duree_ensoleillement)
    print("L'écart type des Duree_ensoleillement est : ",mt.sqrt(variance_Duree_ensoleillement))
    # Affichage de l'histogramme
    plt.hist(Duree_ensoleillement, bins=10, color='orange', edgecolor='black')
    plt.title("Histogramme de la durée d’ensoleillement")
    plt.xlabel("Durée (heures)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

#question5()

#=========================================================================================

# Question 6 :
    
def question6():
    
    covariance_Temperature_minimale_Temperature_maximale = sum(
        (Temperature_minimale[i] - moyenne_Temperature_minimale) * (Temperature_maximale[i] - moyenne_Temperature_maximale)
        for i in range(Nbdonnees)
    ) / Nbdonnees

    covariance_Temperature_minimale_Hauteur_precipitations = sum(
        (Temperature_minimale[i] - moyenne_Temperature_minimale) * (Hauteur_precipitations[i] - moyenne_Hauteur_precipitations)
        for i in range(Nbdonnees)
    ) / Nbdonnees

    covariance_Temperature_minimale_Duree_ensoleillement = sum(
        (Temperature_minimale[i] - moyenne_Temperature_minimale) * (Duree_ensoleillement[i] - moyenne_Duree_ensoleillement)
        for i in range(Nbdonnees)
    ) / Nbdonnees

    covariance_Temperature_maximale_Hauteur_precipitations = sum(
        (Temperature_maximale[i] - moyenne_Temperature_maximale) * (Hauteur_precipitations[i] - moyenne_Hauteur_precipitations)
        for i in range(Nbdonnees)
    ) / Nbdonnees

    covariance_Temperature_maximale_Duree_ensoleillement = sum(
        (Temperature_maximale[i] - moyenne_Temperature_maximale) * (Duree_ensoleillement[i] - moyenne_Duree_ensoleillement)
        for i in range(Nbdonnees)
    ) / Nbdonnees

    covariance_Hauteur_precipitations_Duree_ensoleillement = sum(
        (Hauteur_precipitations[i] - moyenne_Hauteur_precipitations) * (Duree_ensoleillement[i] - moyenne_Duree_ensoleillement)
        for i in range(Nbdonnees)
    ) / Nbdonnees

    corrélation_Temperature_minimale_Temperature_maximale = covariance_Temperature_minimale_Temperature_maximale / (
        mt.sqrt(variance_Temperature_minimale) * mt.sqrt(variance_Temperature_maximale)
    )
    corrélation_Temperature_minimale_Hauteur_precipitations = covariance_Temperature_minimale_Hauteur_precipitations / (
        mt.sqrt(variance_Temperature_minimale) * mt.sqrt(variance_Hauteur_precipitations)
    )
    corrélation_Temperature_minimale_Duree_ensoleillement = covariance_Temperature_minimale_Duree_ensoleillement / (
        mt.sqrt(variance_Temperature_minimale) * mt.sqrt(variance_Duree_ensoleillement)
    )
    corrélation_Temperature_maximale_Hauteur_precipitations = covariance_Temperature_maximale_Hauteur_precipitations / (
        mt.sqrt(variance_Temperature_maximale) * mt.sqrt(variance_Hauteur_precipitations)
    )
    corrélation_Temperature_maximale_Duree_ensoleillement = covariance_Temperature_maximale_Duree_ensoleillement / (
        mt.sqrt(variance_Temperature_maximale) * mt.sqrt(variance_Duree_ensoleillement)
    )
    corrélation_Hauteur_precipitations_Duree_ensoleillement = covariance_Hauteur_precipitations_Duree_ensoleillement / (
        mt.sqrt(variance_Hauteur_precipitations) * mt.sqrt(variance_Duree_ensoleillement)
    )
    return (
        corrélation_Temperature_minimale_Temperature_maximale,
        corrélation_Temperature_minimale_Hauteur_precipitations,
        corrélation_Temperature_minimale_Duree_ensoleillement,
        corrélation_Temperature_maximale_Hauteur_precipitations,
        corrélation_Temperature_maximale_Duree_ensoleillement,
        corrélation_Hauteur_precipitations_Duree_ensoleillement
    )

(
    corrélation_Temperature_minimale_Temperature_maximale,
    corrélation_Temperature_minimale_Hauteur_precipitations,
    corrélation_Temperature_minimale_Duree_ensoleillement,
    corrélation_Temperature_maximale_Hauteur_precipitations,
    corrélation_Temperature_maximale_Duree_ensoleillement,
    corrélation_Hauteur_precipitations_Duree_ensoleillement
) = question6()
    
#  On remarque les variables les plus corélées positivement sont la température maximale et la durée d'ensoleillement
# les plus corélées négativement la hauteur des précipitation et la durée d'ensoleillement
    
# Les moins corélé sont la température minimale et la hauteur des précipitations
    
def afficher_correlations():
   
    # q6a – les plus positivement corrélées
    plt.figure(figsize=(10, 6))
    plt.scatter(Temperature_maximale, Duree_ensoleillement, color='green')
    for i in range(Nbdonnees):
        plt.text(Temperature_maximale[i], Duree_ensoleillement[i], Ville[i], fontsize=8)
    plt.title("q6a - Température max vs Durée d'ensoleillement (corrélation positive)")
    plt.xlabel("Température maximale")
    plt.ylabel("Durée d'ensoleillement")
    plt.tight_layout()
    
    plt.show()

    # q6b – les plus négativement corrélées
    plt.figure(figsize=(10, 6))
    plt.scatter(Hauteur_precipitations, Duree_ensoleillement, color='red')
    for i in range(Nbdonnees):
        plt.text(Hauteur_precipitations[i], Duree_ensoleillement[i], Ville[i], fontsize=8)
    plt.title("q6b - Hauteur précipitations vs Durée d'ensoleillement (corrélation négative)")
    plt.xlabel("Hauteur des précipitations")
    plt.ylabel("Durée d'ensoleillement")
    plt.tight_layout()
    
    plt.show()

    # q6c – les moins corrélées
    plt.figure(figsize=(10, 6))
    plt.scatter(Temperature_minimale, Hauteur_precipitations, color='blue')
    for i in range(Nbdonnees):
        plt.text(Temperature_minimale[i], Hauteur_precipitations[i], Ville[i], fontsize=8)
    plt.title("q6c - Température min vs Hauteur précipitations (faible corrélation)")
    plt.xlabel("Température minimale")
    plt.ylabel("Hauteur des précipitations")
    plt.tight_layout()
    
    plt.show()
  
    
#afficher_correlations() 
    
    
# On regroupe les données de chaque ville
Profils = list(zip(Temperature_minimale, Temperature_maximale, Hauteur_precipitations, Duree_ensoleillement))

# Fonction pour calculer la corrélation de Pearson
def correlation(v1, v2):
    moy1 = sum(v1) / len(v1)
    moy2 = sum(v2) / len(v2)

    num = sum((v1[i] - moy1) * (v2[i] - moy2) for i in range(len(v1)))
    den1 = mt.sqrt(sum((v1[i] - moy1) ** 2 for i in range(len(v1))))
    den2 = mt.sqrt(sum((v2[i] - moy2) ** 2 for i in range(len(v2))))

    if den1 == 0 or den2 == 0:
        return 0
    return num / (den1 * den2)

def question7():

    # Calcul et affichage de la matrice de corrélation
    print("Matrice de corrélation entre les villes :\n")
    for i in range(len(Ville)):
        for j in range(len(Ville)):
            corr = correlation(Profils[i], Profils[j])
            print(f"{Ville[i]} vs {Ville[j]} : {corr:.2f}")
        print()   
    
    
# Calculer la matrice de corrélation
matrice = [[correlation(Profils[i], Profils[j]) for j in range(len(Ville))] for i in range(len(Ville))]

def affichage_corelation():
    question7()
    # Affichage graphique
    plt.figure(figsize=(6, 5))
    plt.imshow(matrice, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label="Corrélation")

    # Ajouter les étiquettes
    plt.xticks(np.arange(len(Ville)), Ville, rotation=45)
    plt.yticks(np.arange(len(Ville)), Ville)
    plt.title("Matrice de corrélation entre les villes")
    plt.tight_layout()
    plt.figure(figsize=(10, 10))

    plt.show()

#affichage_corelation()

#======================================================================================


# Partie e :
    
# Question 15 :
    
    
Annee=[]
Mois=[]
Temperature_maximale2=[]

    
def traitement_de_donnee_fichier2():
    
    
    with open("data2.csv", "r",encoding= "utf-8") as f:
        next(f)
        
        
        for ligne in f:
            meteo=ligne.strip().split(",")
            #print(meteo)
            
            if len(meteo)==3:
            
                Annee.append(float(meteo[0]))
                Mois.append(meteo[1])
                Temperature_maximale2.append(float(meteo[2]))
               
            else:
                print(meteo)
                
    #print(Temperature_maximale)         
    return 
                
        #print(Ville)
        
traitement_de_donnee_fichier2()

def question15():
    
   temperature_2023 = []
   temperature_2024 = []
   mois_2023 = []
   mois_2024 = []

   for i in range(len(Annee)):
       if Annee[i] == 2023:
           temperature_2023.append(Temperature_maximale2[i])
           mois_2023.append(Mois[i])
       elif Annee[i] == 2024:
           temperature_2024.append(Temperature_maximale2[i])
           mois_2024.append(Mois[i])

   plt.figure(figsize=(10, 6))
   plt.plot(mois_2023, temperature_2023, label="2023", color="blue", marker='o')
   plt.plot(mois_2024, temperature_2024, label="2024", color="red", marker='o')
   plt.xlabel("Mois")
   plt.ylabel("Température maximale (°C)")
   plt.title("Évolution des températures maximales à Paris en 2023 et 2024")
   plt.legend()
   plt.grid(True)
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()
    
    
#question15()


"""
# Supposons : Temperature_maximale2 = 12 mois 2023 + 12 mois 2024
X_full = []
y = []

# Construire X et y : prédire mois i à partir de i-1 à i-12
for i in range(12, 24):  # Mois de 2024
    x_i = [Temperature_maximale2[i - k] for k in range(1, 13)]  # mois i-1 à i-12
    X_full.append(x_i)
    y.append(Temperature_maximale2[i])

X_full = np.array(X_full)
y = np.array(y)

meilleur_r2_adj = -float('inf')
meilleure_combinaison = None
meilleur_modele = None

n = len(y)  # 12 mois

# Test de toutes les combinaisons 1 à 11 (évite df_resid = 0)
for k in range(1, 12):
    for indices in itertools.combinations(range(12), k):
        X_subset = X_full[:, indices]
        model = LinearRegression().fit(X_subset, y)
        y_pred = model.predict(X_subset)

        r2 = r2_score(y, y_pred)
        p = len(indices)

        if n - p - 1 == 0:
            continue  # évite division par zéro

        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        if r2_adj > meilleur_r2_adj:
            meilleur_r2_adj = r2_adj
            meilleure_combinaison = indices
            meilleur_modele = model

# Résumé
mois = [f"Mois-{i+1}" for i in range(12)]
mois_choisis = [mois[i] for i in meilleure_combinaison]

print("✅ Meilleur R² ajusté :", round(meilleur_r2_adj, 4))
print("📊 Mois sélectionnés :", mois_choisis)
print("📈 Coefficients :", meilleur_modele.coef_)
print("📍 Interception :", meilleur_modele.intercept_)

# Régression OLS statsmodels
X_best = X_full[:, meilleure_combinaison]
X_best = sm.add_constant(X_best)
model_ols = sm.OLS(y, X_best).fit()
print(model_ols.summary())

"""  
#9 variables sont significatives au seuil de 5%. La variable x4 ne l’est pas, mais le modèle global reste excellent.


#Oui, il existe une relation linéaire significative entre la plupart des variables sélectionnées et la température maximale du mois considéré, au seuil de 5%.
"""    
    


# Construction de X et y : on prédit mois i (mois 12 à 23 = 2024)
# à partir des 12 mois précédents (i-1 à i-12)
X_full = []
y = []

for i in range(12, 24):  # mois 13 à 24 (index 12 à 23), soit 2024
    x_i = [Temperature_maximale2[i - k] for k in range(1, 13)]
    X_full.append(x_i)
    y.append(Temperature_maximale2[i])

X_full = np.array(X_full)
y = np.array(y)

# Recherche de la meilleure combinaison de variables (mois)
meilleur_r2_adj = -float('inf')
meilleure_combinaison = None
meilleur_modele = None

n = len(y)  # Nombre d'observations (12 mois)

for k in range(1, 12):  # on teste toutes les combinaisons de taille 1 à 11
    for indices in itertools.combinations(range(12), k):
        X_subset = X_full[:, indices]
        model = LinearRegression().fit(X_subset, y)
        y_pred = model.predict(X_subset)
        
        r2 = r2_score(y, y_pred)
        p = len(indices)
        
        if n - p - 1 == 0:
            continue  # évite division par zéro pour R2 ajusté
        
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        if r2_adj > meilleur_r2_adj:
            meilleur_r2_adj = r2_adj
            meilleure_combinaison = indices
            meilleur_modele = model

# Affichage des résultats
mois = [f"Mois-{i+1}" for i in range(12)]
mois_choisis = [mois[i] for i in meilleure_combinaison]

print("✅ Meilleur R² ajusté :", round(meilleur_r2_adj, 4))
print("📊 Mois sélectionnés :", mois_choisis)
print("📈 Coefficients :", meilleur_modele.coef_)
print("📍 Interception :", meilleur_modele.intercept_)

# Analyse avec statsmodels pour tests statistiques
X_best = X_full[:, meilleure_combinaison]
X_best = sm.add_constant(X_best)
model_ols = sm.OLS(y, X_best).fit()
print(model_ols.summary())

# --- Prédiction janvier 2025 ---

# Les 12 mois précédents janvier 2025 sont les mois 12 à 23 dans la liste (indices 12 à 23)
X_janv2025 = np.array([Temperature_maximale2[24 - k] for k in range(1, 13)])  # mois 23 à 12
X_janv2025 = X_janv2025[list(meilleure_combinaison)].reshape(1, -1)

prediction_janv2025 = meilleur_modele.predict(X_janv2025)[0]
reelle_janv2025 = 7.5  # température mesurée réelle de janvier 2025
ecart = abs(reelle_janv2025 - prediction_janv2025)

print(f"\n🌡️ Prédiction janvier 2025 : {prediction_janv2025:.2f} °C")
print(f"🌡️ Mesure réelle janvier 2025 : {reelle_janv2025:.2f} °C")
print(f"📏 Écart : {ecart:.2f} °C")

"""

