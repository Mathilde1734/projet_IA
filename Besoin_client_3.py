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

param = pk.load(open('RandomForest_Besoin_client_3.pkl','rb'))


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

def predictions(data, param):
    data_changee = encodage(data, param)
    # print(data_changee)
    predict = pd.DataFrame(param['modele'].predict_proba(data_changee), columns=["proba_inverse", "proba"])
    
    
    # index = data[(data["fk_arb_etat"] == 'SUPPRIMÉ') | 
    #                   (data["fk_arb_etat"]=='ABATTU') | 
    #                   (data["fk_arb_etat"]=='EN PLACE') | 
    #                   (data["fk_arb_etat"]=='REMPLACÉ')].index
    
    # data.drop(index, inplace = True)
    
    data['prediction'] = 0
    index2 = data[(data["fk_arb_etat"] == "Essouché") | (data["fk_arb_etat"] == "Non essouché")].index

    # data.loc[(data["fk_arb_etat"] == "Essouché") | (data["fk_arb_etat"] == "Non essouché"), "prediction"] = predict["proba"]
    j=0
    for i in index2:
        data.loc[i, "prediction"] = predict.loc[j,"proba"]
        j+=1

    # data.loc[index2] = predict["proba"]
    # data["prediction"] = predict['proba']
    # print(data["prediction"])
    return data


def real_carte(data):
    carte = folium.Map(zoom_start=12, location=[49.8476780339,3.2866348474000002])
    colormap = cm.LinearColormap(colors=['green', 'red'], vmin = 0, vmax = 1)
    data = predictions(data,param)
    carte.add_child(colormap)
    # data_reste = data


    # index = data_reste[(data_reste["fk_arb_etat"] == 'Essouché') | (data_reste["fk_arb_etat"]=='Non essouché')].index
    
    # data_reste.drop(index, inplace = True)

    # data = pd.concat([data_prediction, data_reste], axis = 1)
    # print(data["prediction"].value_counts())
    

    for i in range(len(data)):
        folium.Circle(
            location=[data.iloc[i]['latitude'],data.iloc[i]['longitude']],
            radius= (data.iloc[i]['tronc_diam']/3.1415)*0.05 + 1,
            fill = True,
            color = colormap(data.iloc[i]['prediction']),
            # color = 'red' if (data.iloc[i]['prediction'] >= 0.5) else 'green',
            popup= f'<div style="width : 200px">Position de l\'arbre : {data.iloc[i]['latitude'], data.iloc[i]['longitude']}<br>'
                  f'Remarquable : {data.iloc[i]['remarquable']}<br>'
                  f'Stade de développement : {data.iloc[i]['fk_stadedev']}<br>'
                  f'Etat de l\'arbre : {data.iloc[i]['fk_arb_etat']}<br>'
                  f'Quartier : {data.iloc[i]['clc_quartier']}<br>'
                  f'Secteur : {data.iloc[i]['clc_secteur']}<br>'
                  f'Probabilité d\'être déraciné par la tempête : {data.iloc[i]['prediction']}</div>'
 
        ).add_to(carte)


    return carte.save('carte.html')


truc = real_carte(arbre)


