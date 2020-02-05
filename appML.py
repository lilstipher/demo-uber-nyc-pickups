# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Un exemple de visualisation de données géographiques."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import datetime
import pickle


st.title("Commandes des Uber (et autres véhicules de location) dans la ville de New York")
st.markdown(
"""
Il s'agit d'une demo d'une application Streamlit qui montre la répartition géographique des commandes Uber à New York. Utilisez le curseur
pour choisir une heure précise afin de visulaiser les données pour ce moment précis de la journée.

[Lien vers le code source ](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
""")
DATE_TIME = "date/time"
DATA_URL = (
    "data/uber-raw-data-sep14.csv.gz"
)

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data


data = load_data(500000)

hour = st.slider("Heure de la journée", 0, 23)

data = data[data[DATE_TIME].dt.hour == hour]

st.subheader("Données géographiques entre %i:00 et %i:00" % (hour, (hour + 1) % 24))
midpoint = (np.average(data["lat"]), np.average(data["lon"]))

st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data,
            get_position=["lon", "lat"],
            auto_highlight=True,
            radius=100,
            elevation_scale=50,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))

st.subheader("Répartition par minute entre  %i:00 et %i:00" % (hour, (hour + 1) % 24))
filtered = data[
    (data[DATE_TIME].dt.hour >= hour) & (data[DATE_TIME].dt.hour < (hour + 1))
]
hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

st.altair_chart(alt.Chart(chart_data)
    .mark_area(
        interpolate='step-after',
    ).encode(
        x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("pickups:Q"),
        tooltip=['minute', 'pickups']
    ), use_container_width=True)

if st.checkbox("Voir les données brutes", False):
    st.subheader("Données brutes par minute entre %i:00 et %i:00" % (hour, (hour + 1) % 24))
    st.write(data)

### Ajout de modèle de prédiction

st.subheader("Prédition de la demande")
timeOfTheDay= st.time_input("Choisir l'heure", datetime.time(8, 45,00))
st.write('Heure:', timeOfTheDay)
st.markdown("Renseignez vos coordonnées GPS")
## latitude
latitude= st.number_input("Choisir la latitude ", min_value=data['lat'].min(),max_value=data['lat'].max())
longitude= st.number_input("Choisir la longitude ", min_value=data['lon'].min(),max_value=data['lon'].max())

st.write('Vos coordonnées GPS (latitude,longitude) : ', (latitude,longitude))

@st.cache(persist=True)
def load_models():
    model = {}
    nb_pickupsModel=''
    with open('models/kmeansNYCUber.pickle', 'rb') as file:
        model['kmeansNYCUber_model'] = pickle.load(file)

    with open('models/nb_pickupsModel.pickle', 'rb') as file:
        model['nb_pickupsModel'] = pickle.load(file)
    return model
def secondTime(t):
    seconds = (t.hour * 60 + t.minute) * 60 + t.second
    return seconds

model = load_models()
gps = {'lat': [latitude], 'lon': [longitude]}
clusterNo= int(model['kmeansNYCUber_model'].predict(pd.DataFrame(data=gps)))
heureEnSec = secondTime(timeOfTheDay)

time_cluster = {'time': [heureEnSec], 'cluster': [clusterNo]}
pickups = model['nb_pickupsModel'].predict(pd.DataFrame(data=time_cluster))

st.write('La demande prédite dans cette zone à cette heure est de', int(pickups))

