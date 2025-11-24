import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# === Chargement des données et préparation ===
df = pd.read_csv("Breast Cancer Wisconsin.csv")

# Nettoyage
if 'id' in df.columns:
    df.drop(columns='id', inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop(columns='Unnamed: 32', inplace=True)

# Encodage B = 0, M = 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Séparation
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Entraînement
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=10000)
model.fit(X_scaled, y)

# === INTERFACE STREAMLIT ===
st.title("🔬 Diagnostic du Cancer du Sein")
st.write("Veuillez entrer les mesures cliniques d'une tumeur pour obtenir une prédiction.")

# Liste des caractéristiques à saisir
features = list(X.columns)
input_values = []

for feature in features:
    val = st.number_input(f"{feature}", value=float(df[feature].mean()), format="%.5f")
    input_values.append(val)

# Prédiction
if st.button("Prédire"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    # Affichage du résultat
    if prediction == 0:
        st.error("La tumeur est probablement **Maligne**.")
    else:
        st.success("La tumeur est probablement **Bénigne**.")

    # Afficher les probabilités
    proba = model.predict_proba(input_scaled)[0]
    st.markdown("### Probabilités :")
    st.write(f"Probabilité **Maligne** : `{proba[0]*100:.2f}%`")
    st.write(f"Probabilité **Bénigne** : `{proba[1]*100:.2f}%`")


