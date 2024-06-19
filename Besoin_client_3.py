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

# arbre = changement('Données/Data_Arbre.csv','Données/Data_Arbre.json')
arbre = read_json('Données/Data_Arbre.json')

modele = pk.load(open('RandomForest_Besoin_client_3.pkl','rb'))

def encodeur(data):
    new_data = data[["haut_tot","haut_tronc","tronc_diam","fk_arb_etat","fk_stadedev","age_estim", "fk_prec_estim","clc_quartier", "clc_secteur","fk_port","fk_pied","fk_situation","fk_revetement","feuillage"]]

    index = new_data[(new_data["fk_arb_etat"] == 'SUPPRIMÉ') | 
             (new_data["fk_arb_etat"]=='ABATTU') | 
             (new_data["fk_arb_etat"]=='EN PLACE') | 
             (new_data["fk_arb_etat"]=='REMPLACÉ')].index
    new_data.drop(index, inplace = True)

    new_data.loc[new_data["fk_arb_etat"] == "Essouché","fk_arb_etat"] = 1
    new_data.loc[new_data["fk_arb_etat"] != 1,"fk_arb_etat"] = 0
    new_data.fk_arb_etat = new_data.fk_arb_etat.astype(int)

    encodeur = OrdinalEncoder()
    cols = ["clc_quartier", "clc_secteur","fk_port","fk_pied","fk_situation","fk_revetement","feuillage"]
    changement = new_data[cols]
    new_data[cols] = encodeur.fit_transform(changement)

    return new_data


def predictions(data, modele):
    data_encodee = encodeur(data)
    data['prédictions'] = modele.predict(data_encodee[['fk_arb_etat']])
    return data



def real_carte(data):
    carte = folium.Map(zoom_start=12, location=[49.8476780339,3.2866348474000002])
    colormap = cm.LinearColormap(colors=['green', 'red'])
    data = predictions(data,modele)
    

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


