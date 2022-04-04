"""Prototypage de fichier de fonction pour le déploiement, chaque fonction renvoyant un objet JSON contenant
des recommandations d’articles pour un client en particulier, pour l’application responsive sha-cmd/bookshelf"""
from src.objects.CF import CF
from src.objects.CB import lambda_fct
import json
import pandas as pd

from azure import functions
from io import StringIO

 
def inference_keras(req: functions.HttpRequest) -> str:
    """Utilisation d’un neural-network avec Tensorflow via Keras"""
    parameters = req.get_json()
    user = parameters['userId']
    cf = CF()
    return cf.predict(user)


def inference_implicit(req: functions.HttpRequest) -> str:
    """Utilisation de la bibliothèque Implicit"""
    parameters = req.get_json()
    user = parameters['userId']
    df = pd.read_csv('HttpTrigger/similar-artists.tsv', sep='\t', names=['userID', 'itemID', 'est'])
    df['uid'] = df['userID'].map({x: it for it, x in enumerate(df['userID'].unique())})

    myArray = StringIO()
    try:
        recomm = df.loc[df['uid'] == int(user)]['itemID']
        myJSON = [str(x) for x in recomm]
        json.dump(myJSON, myArray)
    except KeyError as e:
        myJSON = []
        json.dump(myJSON, myArray)

    return myArray.getvalue()


def inference_content_based(req: functions.HttpRequest) -> str:
    """Utilisation des tenseurs et de la similarité du cosinus"""
    parameters = req.get_json()
    user = parameters['userId']
    return lambda_fct(user)
