
1. Description du projet

Ce projet a été réalisé dans le cadre du module **Intelligence Artificielle – Machine Learning** à l’Université Ibn Zohr, Faculté des Sciences d’Agadir.  
Il a pour objectif de **prédire si une tumeur mammaire est bénigne ou maligne** à partir du dataset **Breast Cancer Wisconsin (Diagnostic)** en utilisant plusieurs modèles de Machine Learning.


 2. Objectifs
- Explorer et analyser le dataset pour comprendre la distribution des données  
- Prétraiter les données pour les rendre exploitables par les modèles  
- Entraîner et comparer différents modèles de Machine Learning :  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree Classifier  
- Évaluer les performances de chaque modèle et identifier le meilleur  
- Développer une interface utilisateur simple pour interagir avec le modèle 


3. Technologies et Librairies
- **Langage** : Python 3.x  
- **Notebooks** : Jupyter Notebook  
- **Librairies pour le traitement de données** : Pandas, NumPy  
- **Visualisation** : Matplotlib, Seaborn  
- **Machine Learning** : Scikit-Learn  
- **Interface utilisateur** : Streamlit 


4. Structure du projet
Breast-Cancer-Classification/
│
├─ Dataset
├─ .ipynb_checkpoints/ # Scripts Python ( (.ipynb)
├─ models/ # Modèles entraînés (.pkl)
├─ app.py # Interface utilisateur
└─ README.md

5. Instructions pour exécuter le projet

5.1 Cloner le repository

    git clone https://github.com/MARIEMSIBBA/breast-cancer-classification.git
    cd breast-cancer-classification
5.2 Installer les dépendances
     
    pip install -r requirements.txt
    
5.3 Lancer le notebook principal

    jupyter notebook
Ouvrir notebook/breast_cancer_classification.ipynb et 
Suivre les cellules pour explorer, entraîner et évaluer les modèles

5.4 Lancer l’interface 

    python app.py
Ouvrir le lien affiché dans le navigateur ( http://127.0.0.1:5000)

Streamlit :

    streamlit run app.py
L’interface s’ouvrira automatiquement dans le navigateur

6. Auteur
Nom : Mariem Sibba ,  
Université : Université Ibn Zohr, Faculté des Sciences d’Agadir
