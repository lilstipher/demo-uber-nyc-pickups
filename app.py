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

DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

st.title("Commandes des Uber (et autres véhicules de location) dans la ville de New York")
st.markdown(
"""
Il s'agit d'une demo d'une application Streamlit qui montre la répartition géographique des commandes Uber à New York. Utilisez le curseur
pour choisir une heure précise afin de visulaiser les données pour ce moment précis de la journée.

[Lien vers le code source ](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
""")

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data


data = load_data(100000)

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
            radius=100,
            elevation_scale=4,
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
