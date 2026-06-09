import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model + scaler
rf = pickle.load(open('final_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🏀 NBA Shot Success Predictor")
st.markdown("Predict whether an NBA shot will be made based on shot conditions.")

# Sidebar inputs
st.sidebar.header("Shot Parameters")
shot_dist = st.sidebar.slider("Shot Distance (ft)", 0.0, 40.0, 10.0)
close_def_dist = st.sidebar.slider("Closest Defender Distance (ft)", 0.0, 20.0, 4.0)
touch_time = st.sidebar.slider("Touch Time (sec)", 0.0, 10.0, 2.0)
final_margin = st.sidebar.slider("Final Margin", -30, 30, 0)
pts_type = st.sidebar.selectbox("Shot Type", [2, 3])
is_contested = int(close_def_dist < 4)

# Shot zone bin
if shot_dist <= 5:
    zone = 'paint'
elif shot_dist <= 15:
    zone = 'midrange'
elif shot_dist <= 23:
    zone = 'three'
else:
    zone = 'deep'

shot_zone_bin_midrange = int(zone == 'midrange')
shot_zone_bin_three = int(zone == 'three')
shot_zone_bin_deep = int(zone == 'deep')

# Build input df
input_df = pd.DataFrame([{
    'FINAL_MARGIN': final_margin,
    'TOUCH_TIME': touch_time,
    'SHOT_DIST': shot_dist,
    'PTS_TYPE': pts_type,
    'CLOSE_DEF_DIST': close_def_dist,
    'is_contested': is_contested,
    'shot_zone_bin_midrange': shot_zone_bin_midrange,
    'shot_zone_bin_three': shot_zone_bin_three,
    'shot_zone_bin_deep': shot_zone_bin_deep
}])

# Scale
num_cols = ['SHOT_DIST', 'CLOSE_DEF_DIST', 'TOUCH_TIME', 'FINAL_MARGIN']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict
prob = rf.predict_proba(input_df)[0][1]
st.metric("Shot Success Probability", f"{prob:.1%}")

if prob >= 0.5:
    st.success("Model predicts: MADE ✅")
else:
    st.error("Model predicts: MISSED ❌")

# SHAP
st.subheader("Why this prediction?")
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(input_df)
fig, ax = plt.subplots()
shap.waterfall_plot(shap.Explanation(
    values=shap_vals[:,:,1][0],
    base_values=explainer.expected_value[1],
    data=input_df.iloc[0],
    feature_names=input_df.columns.tolist()
), show=False)
st.pyplot(fig)
