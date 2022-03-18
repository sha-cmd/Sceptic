from .src.rec_sys import predict
from .src.objects.RecommenderNet import RecommenderNet
from azure import functions
import json

 
def inference(req: functions.HttpRequest) -> str:
    parameters = req.get_json()
    user = parameters['userId']

    return predict(user)
