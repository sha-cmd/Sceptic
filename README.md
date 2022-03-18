# Sceptic
Recommender Systems
Dans la branche **cosine_similarity** et **contentbasedfilter**, des systèmes de recommandation orientés, respectivement "user" pour la première, et "item" pour la seconde.

Dans la branche **embeddings**, une étude des données, notamment la densité des catégories d’articles, ainsi qu’une étude d’optimisation du taux d’apprentissage de la clusterisation des 
données par plusieurs algorithmes, sans exclure un approfondissement de la technique de clustering avec le manifold T-SNE.

Dans la branche **flask_serverless**, une api pour communiquer avec une application mobile, grâce à un mode déporté dans le cloud nommé Serverless. [AWS Lambda](https://aws.amazon.com/lambda), 
[AzureFunction](https://docs.microsoft.com/en-us/azure/azure-functions/functions-overview) permettent de simplifier la gestion de la fonction en occultant le serveur.

Le fichier [preprocessing](https://github.com/sha-cmd/Sceptic/blob/master/src/preprocessing.py) contient le nettoyage des data. Il produit notamment la base de données database.csv à partir du répertoire de 300 fichiers de clics des clients.

Le fichier [machine learning](https://github.com/sha-cmd/Sceptic/blob/master/src/machine_learning.py) produit tous les tests pour la sélection de nos algorithmes basés sur le collab filtering.

Le fichier [recommender system](https://github.com/sha-cmd/Sceptic/blob/master/src/rec_sys.py) est celui qui permettra de construire la base de données des recommendations pour les clients que nous connaissons assez bien, soit près de 4000, à travers 3000 livres.

De plus les [métriques](https://github.com/sha-cmd/Sceptic/blob/master/src/objects/Metrics.py) sont répertoriées de manière didactique, dans un seul objet.

Nous avons aussi testé l’algorithme par [content based](https://github.com/sha-cmd/Sceptic/blob/master/src/objects/CB.py). 

Pour finir nous avons testé l’algorithme par [collab filter](https://github.com/sha-cmd/Sceptic/blob/master/src/objects/CF.py), dont le réseau neuronal est écrit dans [ce fichier](https://github.com/sha-cmd/Sceptic/blob/master/src/objects/RecommenderNet.py).


