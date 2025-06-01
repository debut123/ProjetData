import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data2.csv")
df_2024 = df[df["Annee"] == 2024]
mois_ordre = [
    "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "novembre", "decembre"
]
df_2024["Mois"] = pd.Categorical(df_2024["Mois"], categories=mois_ordre, ordered=True)
df_2024 = df_2024.sort_values("Mois")

plt.figure(figsize=(10, 5))
plt.plot(df_2024["Mois"], df_2024["Temperature_maximale"], marker='o')
plt.title("Évolution de la température maximale à Paris en 2024")
plt.xlabel("Mois")
plt.ylabel("Température maximale (°C)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
