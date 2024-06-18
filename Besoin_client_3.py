import pandas as pd
import folium

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


def real_carte(data):
    carte = folium.Map(zoom_start=12, location=[49.8476780339,3.2866348474000002])
    
    for i in (len(data)):
        

    return carte.save('carte.html')


truc = real_carte(arbre)


