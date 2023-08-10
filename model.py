import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import webbrowser  # Import the webbrowser module
import subprocess  # Import the subprocess module
import streamlit.components.v1 as components
from streamlit import components
from sklearn.linear_model import LinearRegression
from datetime import datetime

class SessionState:
    def __init__(self):
        self.is_html_open = False
        self.selected_data = None

# Create or get the SessionState
session_state = SessionState()

st.set_page_config(layout="wide")

st.sidebar.title("Detail Analysis for Each Year")

# Add checkbox for data selection
data_options = ["2021", "2022", "2023", "All"]
selected_data_option = st.sidebar.selectbox("Select Data Option", data_options)

if selected_data_option == "All Data":
    session_state.selected_data = data_options[:-1]
else:
    session_state.selected_data = [selected_data_option]


# Add button to open the selected year's HTML page in Chrome
if st.sidebar.button(f"Open detail analyze for {selected_data_option}"):

    path_to_html = f"templates/analyze_{selected_data_option}.html" 

    # Read file and keep in variable
    with open(path_to_html, 'r') as f: 
        html_data = f.read()

    # Update HTML data based on selected data
    # For demonstration purposes, assume that we update the HTML content with the selected data
    # Replace this logic with the appropriate way to modify your HTML content based on data selection
    html_data = html_data.replace("{{DATA_SELECTION}}", ", ".join(session_state.selected_data))

    # Show in webpage
    st.header(f"Detail Analysis for {selected_data_option} data")
    st.components.v1.html(html_data, height=1000, scrolling=True)
    session_state.is_html_open = True

if session_state.is_html_open:
    # Add a button to close the HTML tab
    if st.sidebar.button(f"Close The Detail Analysis for {selected_data_option}"):
        # Clear the iframe by writing an empty string
        components.html("", height=0)
        session_state.is_html_open = False


data = pd.read_excel("data/data.xlsx", header=None, names=["date", "temperature"])
# Convert data into a pandas DataFrame

df = pd.DataFrame(data)

# Convert timestamps to seconds
df['timestamp'] = pd.to_datetime(df['date'], format='%H.%M.%S')
df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

all_data = pd.read_csv("data/all_data_filtered.csv")
all_data['tarih'] = pd.to_datetime(all_data['tarih'], format='%Y/%m/%d %H:%M:%S.%f')

data = pd.DataFrame(df)
# Streamlit app
def main():
    st.title("Mathematical Temperature Prediction App")
    
    st.write("Enter the start and target temperatures below:")

    # User input for start and end temperatures
    start_temp = st.number_input("Start Temperature", value=df['temperature'].min())
    end_temp = st.number_input("Target Temperature", value=df['temperature'].max())

    if st.button("Start Prediction"):

        # Calculate the time taken to reach the target temperature
        #This process is used to calculate the rate of change of temperature values per second.
        data['temperature_increment'] = data['temperature'].diff() / data['seconds'].diff()
        
        # Plot the data and the trend for visualization 
        time_to_reach_end_temp = (end_temp - start_temp) / df['temperature_increment'].mean()

        hours = time_to_reach_end_temp // 3600
        minutes = (time_to_reach_end_temp % 3600) // 60

        st.write(f'Time to reach the end temperature value: {hours} hours, {minutes} minutes.')

        fig = px.line(df, x=data["date"], y=data['temperature'])
        fig.add_hline(y=end_temp, line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)


# Take user input for random start and end temperatures

    st.title("Linear Temperature Prediction App")

    st.write("Enter the start and target temperatures below: (Linear Regression Model)")

    data_linear = pd.read_excel("data/data.xlsx", header=None, names=["date", "temperature"])

    start_temp_linear = st.number_input("Enter the start temperature", value=data_linear["temperature"].min())
    end_temp_linear = st.number_input("Enter the end temperature", value=data_linear["temperature"].max())


    if st.button("Start Linear Prediction"):

        # Convert the 'date' column to datetime format
        data_linear['date'] = pd.to_datetime(data_linear['date'])

        # Calculate the time difference between consecutive timestamps in seconds
        data_linear['time_difference'] = data_linear['date'].diff().dt.total_seconds()

        # Remove the first row with NaN time difference
        data_linear = data_linear.dropna()

        # Calculate the time difference between the first and last timestamps to get the total time elapsed
        total_time_elapsed = (data_linear['date'].iloc[-1] - data_linear['date'].iloc[0]).total_seconds()

        # Calculate the temperature difference between the first and last rows to get the total temperature change
        total_temp_change = data_linear['temperature'].iloc[-1] - data_linear['temperature'].iloc[0]

        # Calculate the average temperature change per second
        average_temp_change_per_second = total_temp_change / total_time_elapsed

        # Create features and labels for the machine learning model
        X = data_linear[['temperature', 'time_difference']]
        y_seconds = (data_linear['temperature'] - data_linear['temperature'].iloc[0]) / average_temp_change_per_second
        y_hours = y_seconds / 3600  # Convert seconds to hours
        y_minutes = y_seconds / 60  # Convert seconds to minutes

        # Train a linear regression model to predict time in seconds
        model_seconds = LinearRegression()
        model_seconds.fit(X, y_seconds)

        # Train a linear regression model to predict time in hours
        model_hours = LinearRegression()
        model_hours.fit(X, y_hours)

        # Train a linear regression model to predict time in minutes
        model_minutes = LinearRegression()
        model_minutes.fit(X, y_minutes)

        # Create a dataframe with user input data
        user_data = pd.DataFrame({
            'temperature': [start_temp_linear, end_temp_linear],
            'time_difference': [10, 10]  # Assuming the user wants predictions for 10 seconds time difference
        })

        # Predict the time in seconds, hours, and minutes
        predicted_seconds = model_seconds.predict(user_data)[1] - model_seconds.predict(user_data)[0]
        predicted_hours = model_hours.predict(user_data)[1] - model_hours.predict(user_data)[0]
        predicted_minutes = model_minutes.predict(user_data)[1] - model_minutes.predict(user_data)[0]

        # Display the result
        st.write(f"Time to reach from {start_temp} degrees to {end_temp} degrees:")
        st.write(f"{int(predicted_hours)} hours and {int(predicted_minutes)} minutes.")

        # Plot the temperature change over time
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data_linear['date'], y=data_linear['temperature'], mode='lines+markers', name='Temperature'))

        fig.update_layout(title='Temperature Change Over Time',
                        xaxis_title='Time',
                        yaxis_title='Temperature (Degrees)',
                        showlegend=True)

        # Add start and end points to the plot
        fig.add_trace(go.Scatter(x=[data_linear['date'].iloc[0], data_linear['date'].iloc[-1]],
                                y=[start_temp_linear, end_temp_linear],
                                mode='markers',
                                marker=dict(size=[10, 10], color=['red', 'green']),
                                name='Start and End Points'))

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
    
    st.title("Time Ranged Temperature Graph")

    start_date = st.date_input("Start Date", value=all_data["tarih"].min())
    end_date = st.date_input("End Date", value=all_data["tarih"].max())


    if st.button("Plot"):

        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
    
        new_data = all_data[(all_data["tarih"]>= start_datetime) & (all_data["tarih"]<= end_datetime)]
        fig2 = go.Figure([go.Scatter(x=new_data['tarih'], y=new_data['Fırın Sıcaklık'])])
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()


