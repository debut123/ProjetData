# -*- coding: utf-8 -*-
"""
Created on Tue May 27 01:03:33 2025

Nom :du chalard

Prénom : François

Projet :data




@author: frdch
"""


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
        
        
#=========================================================================================

# Question 1


          
erreur=traitement_de_donnee()

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
    
    print(" La ville associée à la valeur minimale de durée d’ensoleillement est ",end="")
    print(Ville[Duree_ensoleillement.index(min(Duree_ensoleillement))], end="")
    print(" qui a pour valeur",min(Duree_ensoleillement))
    
    print(" La ville associée à la valeur maximale de durée d’ensoleillement est ",end="")
    print(Ville[Duree_ensoleillement.index(max(Duree_ensoleillement))], end="")
    print(" qui a pour valeur",max(Temperature_minimale)) 
    
    
#question2()

# On remarque que la valeur maximale de la température minimale est inférieur à la
# valeure minimale de la température maximale de plus on constate une grande différence de 
#durée d'ensoleillement qui peux varié de 14.2 à 1363.










