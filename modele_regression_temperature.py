import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("data2.csv")
df_2024 = df[df["Annee"] == 2024].copy()

mois_ordre = [
    "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "novembre", "decembre"
]

df_2024["Mois"] = pd.Categorical(df_2024["Mois"], categories=mois_ordre, ordered=True)
df_2024 = df_2024.sort_values("Mois")
df_2024["NumMois"] = df_2024["Mois"].cat.codes

results = []
for n in range(3, 13):
    subset = df_2024.tail(n)
    X = subset["NumMois"].values.reshape(-1, 1)
    y = subset["Temperature_maximale"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    results.append({
        "n": n,
        "r2": r2,
        "adj_r2": adj_r2,
        "beta0": model.intercept_,
        "beta1": model.coef_[0],
        "model": model
    })

best_model = max(results, key=lambda x: x["adj_r2"])

print("== Meilleur modèle de régression linéaire ==")
print(f"n optimal : {best_model['n']}")
print(f"R²         : {best_model['r2']:.3f}")
print(f"R² ajusté  : {best_model['adj_r2']:.3f}")
print(f"β0 (intercept) : {best_model['beta0']:.2f}")
print(f"β1 (pente)     : {best_model['beta1']:.2f}")

subset = df_2024.tail(best_model['n'])
X = subset["NumMois"].values.reshape(-1, 1)
y = subset["Temperature_maximale"].values
y_pred = best_model["model"].predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Températures réelles')
plt.plot(X, y_pred, color='red', label='Régression linéaire')
plt.title(f"Modèle optimal de régression (n = {best_model['n']})")
plt.xlabel("Numéro du mois (janvier = 0, ..., décembre = 11)")
plt.ylabel("Température maximale (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


mois_janvier = np.array([[0]])
prediction_janvier = best_model["model"].predict(mois_janvier)[0]
temperature_reelle = 7.5
ecart = prediction_janvier - temperature_reelle

print("\n== Prédiction pour janvier 2025 ==")
print(f"Température prédite : {prediction_janvier:.2f} °C")
print(f"Température réelle  : {temperature_reelle:.2f} °C")
print(f"Écart               : {ecart:.2f} °C")

X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
p_value_beta1 = model_sm.pvalues[1]
relation_significative = p_value_beta1 < 0.05

print("\n== Test de significativité de la pente β1 ==")
print(f"p-valeur : {p_value_beta1:.5f}")
print("Relation linéaire significative à 5% :", "Oui" if relation_significative else "Non")