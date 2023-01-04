import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import encoding


from prediction import get_predictions

model = joblib.load(r"model/randomForest2.joblib")
st.set_page_config(
    page_title="Accident Severity Prediction",
    layout='wide'
)

# ['Minutes','Hour','Cause_of_accident','Day_of_week','Number_of_vehicles_involved',
#  'Number_of_casualties','Age_band_of_driver','Type_of_vehicle', 'Light_conditions',
#  'Lanes_or_Medians','Area_accident_occured','Driving_experience']
# creating options for dropdown menu
options_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
                    ' Industrial areas', 'School areas', '  Recreational areas',
                    ' Outside rural areas', ' Hospital areas', '  Market areas',
                    'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
                    'Recreational areas']
options_cause = ['No distancing', 'Changing lane to the right',
                 'Changing lane to the left', 'Driving carelessly',
                 'No priority to vehicle', 'Moving Backward',
                 'No priority to pedestrian', 'Other', 'Overtaking',
                 'Driving under the influence of drugs', 'Driving to the left',
                 'Getting off the vehicle improperly', 'Driving at high speed',
                 'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
                 'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
                        'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
                        'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
                        'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
                 'other', 'Double carriageway (median)', 'One way',
                 'Two-way (divided with solid lines road marking)', 'Unknown']
options_light = ['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit']

feat = ['Minutes','Hours','Cause_of_accident',
        'Day_of_week','Number_of_vehicles_involved',
        'Number_of_casualties','Age_band_of_driver','Type_of_vehicle',
        'Light_conditions','Lanes_or_Medians',
        'Area_accident_occured','Driving_experience']
#features =s
# ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']

st.markdown("<h1 style='text-align: center;'>Accident Severity</h1>",unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):
        st.subheader("Enter input for the required info: ")

        Day_ofWeek = st.selectbox("Select the day: ",options=options_day)
        Minutes = st.slider("Minute: ",0,59,value=0,format="%d")
        Hour = st.slider("Pickup Hour: ",0,23,value=0,format="%d")
        Cause_of_accident = st.selectbox("Select Cause of accident: ",options=options_cause)
        Number_of_vehicles_involved = st.slider("Select the number of vehicles involved: ",1,6,value=0,format="%d")
        Number_of_casualties = st.slider("Select number of casualties: ",1,8,value=0,format="%d")
        Age_band_of_driver = st.selectbox("Select age of driver: ",options=options_age)
        Type_of_vehicle = st.selectbox("Select type of vehicle: ",options=options_vehicle_type)
        Light_conditions = st.selectbox("Select the condition of light: ",options=options_light)
        Lanes_or_Medians = st.selectbox("Select lanes: ",options=options_lanes)
        Area_accident_occured = st.selectbox("Select the area of accident: ",options=options_acc_area)
        Driving_experience = st.selectbox("Select the experience of driver: ",options=options_driver_exp)

        submit = st.form_submit_button("Predict")

        if submit:
            Cause_of_accident = encoding(Cause_of_accident,"Cause_of_accident")
            Day_ofWeek =  encoding(Day_ofWeek,"Day_of_week")
            Age_band_of_driver = encoding(Age_band_of_driver,"Age_band_of_driver")
            Type_of_vehicle = encoding(Type_of_vehicle,"Type_of_vehicle")
            Light_conditions = encoding(Light_conditions,"Light_conditions")
            Lanes_or_Medians = encoding(Lanes_or_Medians,"Lanes_or_Medians")
            Driving_experience = encoding(Driving_experience,"Driving_experience")
            Area_accident_occured = encoding(Area_accident_occured,"Area_accident_occured")
            data = np.array([Minutes,Hour,Cause_of_accident,Day_ofWeek,Number_of_vehicles_involved,Number_of_casualties,Age_band_of_driver,Type_of_vehicle,Light_conditions,Lanes_or_Medians,Area_accident_occured,Driving_experience]).reshape(1,-1)
            pred = get_predictions(data=data,model=model)
            accidentMapping = {2: 'Slight Injury', 1: 'Serious Injury', 0: 'Fatal Injury'}
            st.write(f"Prediction Severity is: {accidentMapping[pred[0]]}")

if __name__ == "__main__":
    main()