import streamlit as st

import Player_Comp

var = 1
print (var)

st.title("NHL Player Comparison Engine")

st.text("Choose a player to compare!")

name_input = st.text_input("Enter full name:")

salary_input = st.number_input ("Enter salary threshold (millions): ")
    
button_true = st.button ("COMPARE")

left, mid, right = st.columns(3)

if (button_true):

    comparison_dict = Player_Comp.comparables(name_input, salary_input)

    with left:
        st.text("Name: " + str.title (comparison_dict["name"][0]))
        st.text('Salary: $' + str(comparison_dict["salary"][0]) + "m")
        st.text('Similarity Score: ' + str(comparison_dict["similar_score"][0]))

    with mid:
        st.text("Name: " + str.title (comparison_dict["name"][1]))
        st.text('Salary: $' + str(comparison_dict["salary"][1]) + "m")
        st.text('Similarity Score: ' + str(comparison_dict["similar_score"][1]))


    with right:
        st.text("Name: " + str.title (comparison_dict["name"][2]))
        st.text('Salary: $' + str(comparison_dict["salary"][2]) + "m")
        st.text('Similarity Score: ' + str(comparison_dict["similar_score"][2]))


reset_button = st.button("RESET")

