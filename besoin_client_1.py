import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def apply_clustering(data, n_clusters):
    # Initialisation du modèle KMeans avec le nombre de clusters spécifié
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Prédiction des clusters et assignation dans une nouvelle colonne 'cluster'
    data['cluster'] = kmeans.fit_predict(data[['haut_tot']])
    
    return data

def detect_anomalies(data):
    # Initialiser le modèle Isolation Forest
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    
    # On ajuste le modèle sur les données des arbres remarquables uniquement
    remarkable_trees = data[data['remarquable'] == 'Oui']
    isolation_forest.fit(remarkable_trees[['haut_tot', 'longitude', 'latitude']])
    
    # Prédire les anomalies pour les arbres remarquables
    data['anomaly'] = 1  # Initialiser comme non-anomalies
    data.loc[data['remarquable'] == 'Oui', 'anomaly'] = isolation_forest.predict(remarkable_trees[['haut_tot', 'longitude', 'latitude']])
    
    # Convertir les résultats (-1 pour les anomalies, 1 pour les points normaux)
    data['anomaly'] = data['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    
    return data

def visualize_tree_clusters(file_path, n_clusters):
    # Chargement des données à partir du fichier CSV
    data = pd.read_csv(file_path, sep=",")
    
    # Sélection des colonnes pertinentes : latitude, longitude, hauteur totale et remarquable
    data = data[['latitude', 'longitude', 'haut_tot', 'remarquable']]
    
    # Suppression des lignes avec des valeurs manquantes
    data.dropna(inplace=True)
    
    # Normalisation de la colonne 'haut_tot' à l'aide de StandardScaler
    scaler = StandardScaler()
    data[['haut_tot']] = scaler.fit_transform(data[['haut_tot']])
    
    # Appliquer le clustering en utilisant la fonction définie précédemment
    data = apply_clustering(data, n_clusters)
    
    # Détecter les anomalies avec Isolation Forest
    data = detect_anomalies(data)
    
    # Évaluation des clusters à l'aide du score de silhouette
    score = silhouette_score(data[['haut_tot']], data['cluster'])
    print(f'Silhouette Score: {score}')
    
    # Ajuster les tailles pour l'affichage sur la carte
    data['size'] = data['haut_tot'] - data['haut_tot'].min() + 1  # Décalage pour avoir des tailles positives
    
    # Création de la figure interactive avec Plotly Express
    fig = px.scatter_mapbox(data, 
                            lat='latitude', 
                            lon='longitude', 
                            color='anomaly',  # Coloration par les anomalies détectées
                            size='size',  # Taille basée sur la hauteur totale normalisée
                            hover_data=['cluster'],  # Informations supplémentaires au survol
                            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},  # Couleurs des points
                            size_max=10,  # Taille maximale des points sur la carte
                            zoom=10,  # Zoom initial de la carte
                            mapbox_style="open-street-map")  # Style de la carte (OpenStreetMap)
    
    fig.show()
    
    return data, scaler

def plot_boxplots(data, scaler):
    # Création d'une figure avec une taille spécifique
    plt.figure(figsize=(12, 8))
    
    # Inversion de la normalisation pour retrouver les valeurs originales de 'haut_tot'
    data['haut_tot'] = scaler.inverse_transform(data[['haut_tot']])
    
    # Tracé du boxplot avec seaborn
    sns.boxplot(x='cluster', y='haut_tot', data=data)
    
    # Ajout de titres et labels pour le graphique
    plt.title('Boxplot des hauteurs totales par cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Hauteur Totale (non normalisée)')
    
    plt.show()  # Affichage du boxplot

def main():
    # Demander à l'utilisateur le nombre de clusters
    n_clusters = int(input("Entrez le nombre de clusters : "))

    file_path = "Données/Data_Arbre.csv"

    # Visualiser les clusters d'arbres
    data, scaler = visualize_tree_clusters(file_path, n_clusters)

    # Visualiser le boxplot des hauteurs totales par cluster
    plot_boxplots(data, scaler)

if __name__ == "__main__":
    main()

