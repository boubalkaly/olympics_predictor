# import all revelant libraries
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

st.title('2024 Olympics Predictor Project 🥇')
st.subheader('Introduction')
'Hi everyone, and thank you for taking the time to read this project! Like the title suggests, we are building a 2024 Olympics Predictor application. Our client seemed really hard at work in the video, so let us help them determine what they should expect if they want to win in the track division'
'For reference, the dataset that I used can be found down below:'
st.markdown('<a href="https://www.kaggle.com/datasets/jayrav13/olympic-track-field-results/data" target="_blank">Results from all Olympic Track & Field Events, 1896 - 2016</a>', unsafe_allow_html=True)
track_field_df = pd.read_csv('track_1896_2016.csv')

# drop any row that has missing values
track_field_df = track_field_df.dropna(subset=['Result'])
track_field_df = track_field_df.drop(columns=['Wind'])


'Before we move on, let us take a quick glance at the dataset:'

st.title('Track and Field Results Tables 1896-2024')
st.dataframe(track_field_df)


'We are going to perform some data cleaning first. Since the wind column has mostly NaN values, we are going to remove it from the dataset. We are also going to drop any row that contains missing values. Afterwards, we are going to add a results column in seconds to simplify our data visualization later down the line!'
''
# drop any row that has missing values
track_field_df = track_field_df.dropna(subset=['Result'])

selected_events = ['10000M Men', '5000M Men',
                   '800M Men', '3000M Steeplechase Men']
events = set()

for event in track_field_df['Event']:
    if 'Women' not in event:
        if event in selected_events:
            events.add(event)
# track_field_df = track_field_df.drop(columns=['Wind'])
event = st.selectbox(
    'Select the event you are interested in',
    events)

'You selected: ', event

filtered_df = track_field_df.loc[track_field_df['Event'] == f'{event}']


for time in filtered_df['Result']:

    if 'h' in time:
        hours, remainder = time.split('h')
        hours = int(hours)
        time = pd.to_timedelta(f'{hours}:{remainder}')
        continue
    if '-' in time:
        hours, remainder = time.split('-')
        hours = int(hours)
        time = pd.to_timedelta(f'{hours}:{remainder}')
        continue

    parts = time.split(':')

    if len(parts) == 2:
        # If there are 2 parts (MM:SS.SS), append '00' for hours and convert
        time = '0:' + time
    elif len(parts) == 3:
        # If there are 3 parts (HH:MM:SS), just convert
        pass
    else:
        raise ValueError(f"Unexpected time format: {time}")
    time = pd.to_timedelta(time)
filtered_df.loc[:, 'Result'] = '0:' + filtered_df['Result']
filtered_df['Result'] = pd.to_timedelta(filtered_df['Result'], errors='coerce')


# def custom_to_timedelta(time_str):
#     if 'h' in time_str:
#         hours, remainder = time_str.split('h')
#         hours = int(hours)
#         return pd.to_timedelta(f'{hours}:{remainder}')

#     parts = time_str.split(':')

#     if len(parts) == 2:
#         # If there are 2 parts (MM:SS.SS), append '00' for hours and convert
#         time_str = ('0:' + time_str).split('.')[0]
#     elif len(parts) == 3:
#         # If there are 3 parts (HH:MM:SS), just convert
#         pass
#     else:
#         raise ValueError(f"Unexpected time format: {time_str}")

#     return pd.to_timedelta(time_str)


# filtered_df['Result'] = filtered_df['Result'].apply(custom_to_timedelta)

filtered_df.loc[:,
                'Result in Seconds'] = filtered_df['Result'].dt.total_seconds()
st.dataframe(filtered_df)
st.title('Visualizing Track and Field Results in Seconds over the Years')

sn.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sn.scatterplot(data=filtered_df, x='Year', y='Result in Seconds', marker='o')
plt.title('Scatter plot of Years vs. Time in Seconds')
plt.xlabel('Competition Year')
plt.ylabel('Results in Seconds')

# Display the plot in Streamlit
st.subheader('Scatter plot of the data')
st.pyplot(plt)

y = filtered_df['Result in Seconds'].values
X = filtered_df['Year'].values

X = X.reshape(-1, 1)


# we are going to start by linear regression analysis and then do a Ridge regression

# we are using a 20/80 partition for training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Using Linear Regression for the prediction
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

sn.set(style="whitegrid")
sn.scatterplot(x=x_test.flatten(), y=y_test, marker='o')
sn.lineplot(x=x_test.flatten(), y=predictions, color='red')
plt.title('Predictions with Ridge Regression')
plt.xlabel('Competition Year')
plt.ylabel('Results in Seconds')

'After performing linear regresssion with a 20/80 test/train split, we get the following plot:'
st.subheader('Linear Regression Graph')
st.pyplot(plt)

years = list(range(2020, 2025, 4))

# Create the DataFrame
years_df = pd.DataFrame(years, columns=['Year'])

selected_year = st.selectbox(
    'Select the year you want to predict for using linear Regression',
    years_df['Year'], key='linear')

'You selected: ', selected_year


selected_year = np.array([selected_year]).reshape(-1, 1)

year_prediction = model.predict(selected_year)


'If you want a medal, your performance should be at least (in seconds): ', year_prediction


st.subheader('Using Ridge Regression vs. Linear Regression')


# we are adding a standard scaler and polynomial features to try to better fit the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)


model = Ridge(alpha=1.0)
model.fit(x_train_poly, y_train)
predictions = model.predict(x_test_poly)

# print(predictions)

sn.set(style="whitegrid")
sn.scatterplot(x=x_test.flatten(), y=y_test, marker='o')
sn.lineplot(x=x_test.flatten(), y=predictions, color='blue')
plt.title('Predictions with Ridge Regression')
plt.xlabel('Competition Year')
plt.ylabel('Results in Seconds')

st.pyplot(plt)

selected_year = st.selectbox(
    'Select the year you want to predict for using linear Regression',
    years_df['Year'], key='ridge')

'You selected: ', selected_year


selected_year = np.array([selected_year]).reshape(-1, 1)

selected_year_scaled = scaler.fit_transform(selected_year)
selected_year_poly = poly.fit_transform(selected_year_scaled)

year_prediction = model.predict(selected_year_poly)


'If you want a medal, your performance should be at least (in seconds): ', year_prediction


st.subheader('Conclusion and observations')
'There we go. This is how you can predict how much you need to train and what results you should get yourself in you want a win! In the future, it would be good to predict the outcomes using other models such as a decision tree, should more features for the data be available'
