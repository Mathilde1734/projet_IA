# projet_IA 
## Besoin client 3

### Notebook
Le fichier Besoin_client_3.ipynb permet d'entraîner le modèle et de l'exporter une fois que les performances sont satisfaisantes.
Il prend donc un fichier cvs comme base de données et renvoi un encodeur et un modèle  opérationnel. Il prend comme paramètre, notre base de données nettoyées pour l'entraînement.

### Script Python
Ce ficher Besoin_client_3.py permet d'afficher la carte en fonction des prédictions faites par le modèle. Il prend donc une base de données à encoder, l'encodeur et le modèle du Notebook. Une fois exécuté, il renvoi une carte qui affiche la probabilité des arbres en fonction de leur état si ils seront arrachés ou non lors d'une prochaine tempête. Il prend comme paramètre, la base de données fournit par nos professeurs.

### Fichier .pkl
Le fichier RandomForest_Besoin_client_3.pkl est un fichier qui "sauvegarde" les données de l'encodeur et du modèle une fois entraîné.