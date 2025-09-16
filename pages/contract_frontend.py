import streamlit as st
import  contract_prediction 

st.title("ML-Based NHL Contract Predictor")

st.text("This model has been trained on 500+ players & 40+ advanced metrics to provide accurate contract valuations!")

st.text("Please enter the following 15 advanced metrics to obtain an accurate projection of the player's contract:")

with st.form ("projection_stats"):

    one, two, three = st.columns(3)

    with one: 
        age = st.number_input("Age")
        pts_60 = st.number_input ("PTS/60")
        ppg_gp = st.number_input ("PPG/GP")
        oiSH_percent = st.number_input ("oiSH%")
        ff_60 = st.number_input ("FF/60")

    with two:
        pp_gp = st.number_input ("PP/GP")
        blk_60 = st.number_input ("BLK/60")
        oiSV_percent = st.number_input ("oiSV%")
        e_plus_minus = st.number_input ("E+/-")
        give_60 = st.number_input ("GIVE/60")

    with three:
        cf_60 = st.number_input ("CF/60")
        plus_minus = st.number_input ("+/-")
        cf_percent_rel = st.number_input ("CF%" + " rel")
        fa_60 = st.number_input ("FA/60")
        tsa_60 = st.number_input ("TSA/60")


    submission = st.form_submit_button("Obtain Contract Projection")

if submission:
    projection = contract_prediction.contract_predictor(pp_gp, age, oiSH_percent, pts_60, ff_60, ppg_gp, blk_60, oiSV_percent, e_plus_minus, give_60, cf_60, plus_minus, cf_percent_rel, fa_60, tsa_60)
    st.write(("The contract prediction for this player is $" + str(projection[0].round(3)) +  " million!"))     


st.write ("The top 15 stats account for 86 percent of the model's accuracy. Therefore, they can be used to make an accurate prediction, instead of using all 40+ features.")

st.write("The picture below shows a plot summarizing the importance of each feature using SHAP: ")

st.image("SHAP_nhl.png", use_container_width=True)

