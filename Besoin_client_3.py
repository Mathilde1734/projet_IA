import pandas as pd
import pickle as pk
import folium
from sklearn.preprocessing import OrdinalEncoder
import branca.colormap as cm

def transformation(df, chemin_json):
    df.to_json(chemin_json)

def read_json(chemin_json):
    data_json = pd.read_json(chemin_json)
    return data_json

def changement(chemin_csv, chemin_json):
    df = pd.read_csv(chemin_csv)
    transformation(df, chemin_json)
    data = read_json(chemin_json)
    return data

arbre = changement('Données/Data_Arbre.csv','Données/Data_Arbre.json')
# arbre = read_json('Données/Data_Arbre.json')

param = pk.load(open('RandomForest_Besoin_client_3.pkl','rb'))


def encodage(data,param):
    data_encodee = param['encodeur'].transform(data.drop('fk_arb_etat', axis=1))
    return data_encodee

def predictions(data, param):
    data_changee = encodage(data, param)
    data['prédictions'] = param['modele'].predict(data_changee)
    return data



def real_carte(data):
    carte = folium.Map(zoom_start=12, location=[49.8476780339,3.2866348474000002])
    colormap = cm.LinearColormap(colors=['green', 'red'])
    data = predictions(data,param)
    

    for i in (len(data)):
        folium.Circle(
            location=[data.iloc[i]['latitude'],data.iloc[i]['longitude']],
            radius= (data.iloc[i]['tronc_diam']/3.1415)*0.05 + 1,
            color = colormap(data.iloc[i]['prédictions']) ,
            popup= f'<div style="width : 200px">Position de l\'arbre : {data.iloc[i]['latitude'], data.iloc[i]['longitude']}<br>'
                  f'Remarquable : {data.iloc[i]['remarquable']}<br>'
                  f'Stade de développement : {data.iloc[i]['fk_stadedev']}<br>'
                  f'Etat de l\'arbre : {data.iloc[i]['fk_arb_etat']}<br>'
                  f'Quartier : {data.iloc[i]['clc_quartier']}<br>'
                  f'Secteur : {data.iloc[i]['clc_secteur']}</div>'
 
        ).add_to(carte)


    return carte.save('carte.html')


truc = real_carte(arbre)


