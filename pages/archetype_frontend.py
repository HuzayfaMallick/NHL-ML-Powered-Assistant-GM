import streamlit as st
from rapidfuzz import process

st.title("ML-Powered NHL Player Archetype Analysis")

player_list = []

st.text("This model has been trained using KMeans clustering to identify players that are a similar archetype. This tool can be used by general managers to look for suitable replacement players in free agency to help with roster construction.")

st.text("Below, you will be able to enter the name of a player and receive their archetype relative to the rest of the league.")

input = st.text_input("Enter player name: ")

if input: 
    match = process.extractOne(input, player_list)
    st.write("Closest match: " + match[0])
    player = match[0]
