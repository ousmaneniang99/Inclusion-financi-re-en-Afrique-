#Importation des librairies python
import streamlit as st
import pandas as pd
import numpy as np
#from ydata_profiling import ProfileReport
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import missingno as msno
from sklearn.linear_model import LogisticRegression


#importation du dataset
data=pd.read_csv('Financial_inclusion_dataset.csv')

#data head()
data.head()

# information générale
data.info()

data.isnull().sum()

data.describe().T

#copions le dataset d'origine dans une variable df 
df=data.copy()


#voire les valeurs de notre variable cible
df.bank_account.unique()

# voyons la repartition des deux types de personnes
#%matplotlib inline
plt.xlabel('compte')
plt.ylabel('nombre de personnes')
plt.bar(df.bank_account, height=0.3)
plt.show()

#affichages valeurs des colonnes catégorielles
for column in df.select_dtypes(include=['object']).columns:
    unique_values = df[column].unique()
    print(f"Valeurs uniques de la colonne '{column}': {unique_values}")

#effacer les colonnes unitile pour notre analyse
df=df.drop(columns=['uniqueid','year'],axis=1)


# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

# Encoder chaque colonne de type object
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])


# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

# Encoder chaque colonne de type object
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])


# choix du features  et de la variable cible 
X = df.drop(columns=['bank_account'], axis=1)
y = df['bank_account']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Entraînement le modèle
model.fit(X_train, y_train)

# prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


# Charger le modèle entraîné
# Sauvegarder le modèle entraîné
#joblib.dump(model, 'model.pkl')
#model = joblib.load('model.pkl')

# Titre de l'application
st.title("Prédiction")

# Créer des champs de saisie pour les features
country = st.selectbox("Pays", df['country'].unique())
location_type = st.selectbox("Type de location: urbain ou rural", df['location_type'].unique())
household_size = st.number_input("surface de la maison", min_value=0)
age_of_respondent = st.number_input("âge", min_value=0)
cellphone_access = st.selectbox("Accés telephone", df['cellphone_access'].unique())
marital_status=st.selectbox('situation matrimoniale',df['marital_status'].unique())
gender_of_respondent=st.selectbox('genre',df['gender_of_respondent'].unique())
relationship_with_head=st.selectbox('statut dans la famille',df['relationship_with_head'].unique())
education_level=st.selectbox('Niveau etude',df['education_level'].unique())
job_type=st.selectbox('Titre de titre',df['job_type'].unique())


# Bouton pour déclencher la prédiction
if st.button("Prédire"):
    # Créer un DataFrame avec les valeurs saisies
    input_data = pd.DataFrame({
        'job_type': [job_type],
        'country': [country],
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    })

    # Faire la prédiction
    prediction = model.predict(input_data)[0]

    # Afficher le résultat
    if prediction == 0:
        st.success("personnes n'est pas susceptibles d'avoir ou d'utiliser un compte bancaire")
    else:
        st.warning("la personne est  susceptible d'avoir ou d'utiliser un compte bancaire.")
