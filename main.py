"""
streamlit run main.py [-- script args]
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pandas as pd


header=st.container()
dataset=st.container()
features=st.container()
model_training=st.container()

st.markdown("""
<style>
.main { background-color: #f5f5f5; }
</style>
""",unsafe_allow_html=True)

@st.cache_data
def get_data():
    taxi_data=pd.read_csv('data/taxi_tripdata.csv', dtype={'store_and_fwd_flag': 'str'})
    return taxi_data

with header:
    st.title('welcome to my first data science project')
    st.text('In this project i look into the transactions')

with dataset:
    st.header('Bank Transaction dataset')
    st.text('I found this dataset on kaggle')

  
    # Load taxi data
    taxi_data=get_data()
    st.write(taxi_data.head(5))

    pulocation_dist=taxi_data['PULocationID'].value_counts()
    st.subheader('Pickup location distribution')
    st.bar_chart(pulocation_dist)

with features:
    st.header('The features i created')
    st.markdown('* **first feature:** I created this feature because of this...')
    st.markdown('* **second feature:** I created this feature because of this...')
    st.markdown('* **third feature:** I created this feature because of this...')

with model_training:
    st.header('Time to train the model')
    st.text('Here you can choose the hyperparameters of the model and see how the performance changes')
    sel_col,disp_col=st.columns(2)

    max_depth=sel_col.slider('What should be the max_depth of the model?',min_value=10,max_value=100,value=20,step=10)
    n_estimators=sel_col.selectbox('How many trees should there be?',options=[100,200,300,'No limit'],index=0)

    sel_col.text('Here is the list of features in my data:')
    sel_col.write(taxi_data.columns)

    input_feature=sel_col.text_input('Which feature should be used as the input feature?','PULocationID')

    if n_estimators=='No limit':
        regr=RandomForestRegressor(max_depth=max_depth,random_state=0)
    else:
        regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators,random_state=0)

    # split the data to train and test sets using train_test_split
    x_train,x_test,y_train,y_test=train_test_split(taxi_data[[input_feature]],taxi_data[['trip_distance']],test_size=0.3,random_state=0)

    regr.fit(x_train,y_train)
    prediction=regr.predict(x_test)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y_test,prediction))
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y_test,prediction))
    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y_test,prediction))
