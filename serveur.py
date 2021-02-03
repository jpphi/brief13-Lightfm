#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:23:45 2021

@author: jpphi
"""
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from time import time

from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset

from scipy.sparse import csr_matrix



app = Flask(__name__)

plays = pd.read_csv('datasets/user_artists.dat', sep='\t')
artists = pd.read_csv('datasets/artists.dat', sep='\t', usecols=['id','name'])

# Merge (fusionner) artist and user pref data
ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
ap = ap.rename(columns={"weight": "playCount"})

# Group artist by name
artist_rank = ap.groupby(['name']) \
    .agg({'userID' : 'count', 'playCount' : 'sum'}) \
    .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
    .sort_values(['totalPlays'], ascending=False)

artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']
#print(artist_rank)
artist_names= ap["name"].unique()
artist_names.sort()
artist_names= tuple(artist_names)
#print(ap.columns)
#print(ap)

# Merge into ap matrix
ap = ap.join(artist_rank, on="name", how="inner") \
    .sort_values(['playCount'], ascending=False)

# Preprocessing
pc = ap.playCount
play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
ap = ap.assign(playCountScaled=play_count_scaled)
#print(ap)

# Build a user-artist rating matrix 
ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
ratings = ratings_df.fillna(0).values
#print(ratings.shape)
vecteur=np.zeros(ratings.shape[1])


@app.route('/', methods=['GET','POST'])
def dashboard():

    noms= request.form.getlist("dblst_artists")
    sugg= []
    #print(noms)

    for el in noms:
        artiste= ap[ap.name== el]
        lind= list(artiste.artistID)[0] -1
        vecteur[lind]= artiste.playCountScaled.median()

    X= np.vstack((ratings,vecteur))
    n_users, n_items = X.shape


    Xcsr = csr_matrix(X)
    Xcoo = Xcsr.tocoo()
    data = Dataset()
    data.fit(np.arange(n_users), np.arange(n_items))
    interactions, weights = data.build_interactions(zip(Xcoo.row, Xcoo.col, Xcoo.data)) 
    train, test = random_train_test_split(interactions)

    model = LightFM(learning_rate=0.05, loss='warp')
    model.fit(train, epochs=10, num_threads=2)

    scores = model.predict(0, vecteur)
    print("score:",scores)
    top_items = ap["name"].unique()[np.argsort(-scores)]

    sugg= top_items[:10]
    

    """
    print(top_items)

    #print(X)
    """

    return render_template("page.html", artist_names= artist_names, noms= noms, sugg= sugg)

if __name__ == "__main__":
    app.run(debug=True, port= 5000)