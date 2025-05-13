# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("player_stats.csv")

df = load_data()

# Drop unnecessary columns
df_numeric = df.drop(columns=['Player', 'Season', 'Team'], errors='ignore')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to original DataFrame
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Sidebar: Player selection
player_name = st.selectbox("Select a player:", df['Player'])

# Get the selected player's PCA coordinates
selected_coords = df[df['Player'] == player_name][['PCA1', 'PCA2']].values[0]

# Compute Euclidean distance to all players
df['Distance'] = np.linalg.norm(df[['PCA1', 'PCA2']].values - selected_coords, axis=1)

# Get top 5 most similar players (excluding the selected player)
similar_players = df[df['Player'] != player_name].nsmallest(5, 'Distance')

# Display results
st.subheader("Top 5 Most Similar Players")
st.table(similar_players[['Player', 'Team', 'Season', 'Distance']])

# PCA scatter plot
fig = px.scatter(
    df,
    x='PCA1',
    y='PCA2',
    hover_name='Player',
    color=df['Player'].apply(lambda x: 'Selected' if x == player_name else ('Similar' if x in similar_players['Player'].values else 'Others')),
    color_discrete_map={
        'Selected': 'red',
        'Similar': 'orange',
        'Others': 'lightgray'
    },
    title="PCA Plot of Players"
)

st.plotly_chart(fig)
