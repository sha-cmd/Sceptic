import json
import pandas as pd
import os

from azure.storage.blob import BlobServiceClient
from azure import functions
from io import StringIO
from io import BytesIO

global df  # data tsv
CONNECT_STR = "DefaultEndpointsProtocol=https;AccountName=scepticist;AccountKey=VXpbGMMx0vptbpvcSVce9WZN5uLogeBEZGLFHk0M9EH3aWVo+twhux3YO0QgsWUUJSAh+ouQXnfXIsx7HU97dw==;EndpointSuffix=core.windows.net"
CONTAINERNAME= 'shrek'
BLOBNAME= 'similar-artists.tsv'

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)

# Create the container
container_client = blob_service_client.get_container_client(CONTAINERNAME)

# Download the blob
blobstring = container_client.download_blob(BLOBNAME)
inmemoryfile = BytesIO(blobstring.readall())

df = pd.read_csv(inmemoryfile, sep='\t', names=['userID', 'itemID', 'est'])
#df = pd.read_csv('HttpTrigger/' + LOCALFILENAME, sep='\t', names=['userID', 'itemID', 'est'])
df['uid'] = df['userID'].map({x: it for it, x in enumerate(df['userID'].unique())})

 
def inference(req: functions.HttpRequest) -> str:
    global df
    parameters = req.get_json()
    user = parameters['userId'] 
    
    try:
        recomm = df.loc[df["uid"] == int(user)]['itemID']
        myJSON = [str(x) for x in recomm]
        myArray = StringIO()
        json.dump(myJSON, myArray)
        return myArray.getvalue() 
    except KeyError as e:
        myJSON = []
        myArray = StringIO()
        json.dump(myJSON, myArray)
        return myArray.getvalue()


