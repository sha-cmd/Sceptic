from src.objects.CF import CF

import json
import pandas as pd

from azure import functions
from io import StringIO

 
def inference_keras(req: functions.HttpRequest) -> str:
    parameters = req.get_json()
    user = parameters['userId']
    cf = CF()
    return cf.predict(user)


def inference_implicit(req: functions.HttpRequest) -> str:
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
