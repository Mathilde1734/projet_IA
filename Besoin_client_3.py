import pandas as pd
import pickle as pk
import folium
from sklearn.preprocessing import OrdinalEncoder
import branca.colormap as cm
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Tranformation d'un dataFrame en format json
def transformation(df, chemin_json):
    df.to_json(chemin_json)

# Lecture du dataFrame en json
def read_json(chemin_json):
    data_json = pd.read_json(chemin_json)
    return data_json

# Prend un fichier csv et le transforme en fichier json
def changement(chemin_csv, chemin_json):
    df = pd.read_csv(chemin_csv)
    transformation(df, chemin_json)
    data = read_json(chemin_json)
    return data

# Importer les données du Web
def importation():
    url = 'https://www.infoclimat.fr/observations-meteo/temps-reel/saint-quentin-roupy/07061.html'
    # url = 'https://fr.weatherspark.com/y/49517/M%C3%A9t%C3%A9o-moyenne-%C3%A0-Saint-Quentin-France-tout-au-long-de-l\'ann%C3%A9e#Figures-Rainfall'
    # url = 'https://fr.weatherspark.com/y/49517/M%C3%A9t%C3%A9o-moyenne-%C3%A0-Saint-Quentin-France-tout-au-long-de-l\'ann%C3%A9e#Figures-Temperature'
    req = requests.get(url)
    soup = BeautifulSoup(req.text)
    meteo = soup.find_all("table")[0]
    df_meteo = pd.read_html(str(meteo))[0]
    export_csv = df_meteo.to_csv(r'Données/données_météo.csv',index=None, header=True)


# Requête API pour récupérer des données
def API():
    url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Saint-quentin/today?unitGroup=metric&key=N64WMQKA5BRGE5FV2QJFETR6H&contentType=json'
    table = requests.get(url).json()
    return table['days'][0]['humidity']


def prob_himidite(data,humidity):
    data['Humidité'] = data['prediction']
    if(humidity > 50) :
        data.loc[data['fk_pied']=='Terre', 'Humidité'] = 1
    
    return data


# Encoder les données json de manière a ce que le modèle s'y retrouve
def encodage(data,param):
    data_prep=data.copy()

    index = data_prep[(data_prep["fk_arb_etat"] == 'SUPPRIMÉ') | 
                      (data_prep["fk_arb_etat"]=='ABATTU') | 
                      (data_prep["fk_arb_etat"]=='EN PLACE') | 
                      (data_prep["fk_arb_etat"]=='REMPLACÉ')].index
    
    data_prep.drop(index, inplace = True)

    data_reduit = data_prep[["haut_tot","haut_tronc","tronc_diam","age_estim", "fk_prec_estim","fk_pied","fk_situation"]]
    columns_enc=["fk_pied","fk_situation"]
    data_reduit[columns_enc] = param['encodeur'].transform(data_reduit[columns_enc])

    return data_reduit


# Fonction de prédiction du modèle sur les données qu'ils reçoient
def predictions(data, param):
    data_changee = encodage(data, param)
    predict = pd.DataFrame(param['modele'].predict_proba(data_changee), columns=["proba_inverse", "proba"])
    
    data['prediction'] = 0
    index2 = data[(data["fk_arb_etat"] == "Essouché") | (data["fk_arb_etat"] == "Non essouché")].index

    j=0
    for i in index2:
        data.loc[i, "prediction"] = predict.loc[j,"proba"]
        j+=1

    return data

# Affichage de la carte en fonction des prédictions faites par le modèle
def real_carte(data):
    carte = folium.Map(zoom_start=12, location=[49.8476780339,3.2866348474000002])
    colormap = cm.LinearColormap(colors=['green', 'red'], vmin = 0, vmax = 1)
    data = predictions(data,param)
    carte.add_child(colormap)

    humidity = API()
    data = prob_himidite(data, humidity)

    case1 = folium.FeatureGroup(name='Prédiction des arbres déracinés', show=True).add_to(carte)
    case2 = folium.FeatureGroup(name='Prédiction météorologiques', show=False).add_to(carte)
   
    for i in range(len(data)):
        folium.Circle(
            location=[data.iloc[i]['latitude'],data.iloc[i]['longitude']],
            radius= (data.iloc[i]['tronc_diam']/3.1415)*0.05 + 1,
            fill = True,
            color = colormap(data.iloc[i]['prediction']),
            popup= f'<div style="width : 200px">Position de l\'arbre : {data.iloc[i]['latitude'], data.iloc[i]['longitude']}<br>'
                  f'Remarquable : {data.iloc[i]['remarquable']}<br>'
                  f'Stade de développement : {data.iloc[i]['fk_stadedev']}<br>'
                  f'Etat de l\'arbre : {data.iloc[i]['fk_arb_etat']}<br>'
                  f'Quartier : {data.iloc[i]['clc_quartier']}<br>'
                  f'Secteur : {data.iloc[i]['clc_secteur']}<br>'
                  f'Probabilité d\'être déraciné par la tempête : {data.iloc[i]['prediction']}</div>'
 
        ).add_to(case1)

        folium.Circle(
            location=[data.iloc[i]['latitude'],data.iloc[i]['longitude']],
            radius= (data.iloc[i]['tronc_diam']/3.1415)*0.05 + 1,
            fill = True,
            color = colormap(data.iloc[i]['Humidité']),
            popup= f'<div style="width : 200px">Position de l\'arbre : {data.iloc[i]['latitude'], data.iloc[i]['longitude']}<br>'
                  f'Remarquable : {data.iloc[i]['remarquable']}<br>'
                  f'Stade de développement : {data.iloc[i]['fk_stadedev']}<br>'
                  f'Etat de l\'arbre : {data.iloc[i]['fk_arb_etat']}<br>'
                  f'Quartier : {data.iloc[i]['clc_quartier']}<br>'
                  f'Secteur : {data.iloc[i]['clc_secteur']}<br>'
                  f'Probabilité d\'être déraciné par la tempête : {data.iloc[i]['Humidité']}</div>'
 
        ).add_to(case2)

    folium.LayerControl().add_to(carte)


    return carte.save('carte.html')



if __name__ =="__main__":

    arbre = changement('Données/Data_Arbre.csv','Données/Data_Arbre.json')
    meteo = changement('Données/données_météo.csv','Données/données_météo.json')
    param = pk.load(open('RandomForest_Besoin_client_3.pkl','rb'))

    importation()

    # Affichage de la carte
    truc = real_carte(arbre)
