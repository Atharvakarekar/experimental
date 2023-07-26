# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
#
# # Synthetic dataset: Replace this with real dyslexia data
# # Each row represents a sample, and the columns represent different features
# # The target variable 'label' represents whether the individual has dyslexia (1) or not (0)
# data = {
#     'feature1': [1.5, 2.0, 1.0, 2.5, 3.0, 2.7, 4.0, 3.2, 4.5, 3.8],
#     'feature2': [2.5, 3.0, 2.0, 3.5, 4.0, 3.7, 5.0, 4.2, 5.5, 4.8],
#     'feature3': [3.5, 4.0, 3.0, 4.5, 5.0, 4.7, 6.0, 5.2, 6.5, 5.8],
#     'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 0: Non-dyslexic, 1: Dyslexic
# }
#
# # Convert the data to NumPy arrays
# X = np.array([data['feature1'], data['feature2'], data['feature3']]).T
# y = np.array(data['label'])
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize the Support Vector Machine (SVM) classifier
# classifier = SVC(kernel='linear')
#
# # Train the classifier on the training data
# classifier.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = classifier.predict(X_test)
#
# # Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# # Display classification report (includes precision, recall, F1-score, etc.)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
from http import client

# import streamlit as st
#
# # Assuming you have a custom Connection for the Weather API
# from your_weather_api_connection import WeatherAPIConnection
#
# def main():
#     st.title("Weather App")
#
#     # Assuming the WeatherAPIConnection provides a method to get weather data
#     weather_connection = WeatherAPIConnection()
#
#     # Get user input for location (city or zip code)
#     location = st.text_input("Enter Location (City or Zip Code):")
#
#     if st.button("Get Weather"):
#         if location:
#             weather_data = weather_connection.get_weather(location)
#
#             if weather_data:
#                 st.subheader(f"Weather in {location}:")
#                 st.write(f"Temperature: {weather_data['temperature']} °C")
#                 st.write(f"Conditions: {weather_data['conditions']}")
#                 st.write(f"Humidity: {weather_data['humidity']}%")
#             else:
#                 st.write("Unable to fetch weather data. Please check your location.")
#         else:
#             st.warning("Please enter a location.")
#
# if __name__ == "__main__":
#     main()



# import pandas as pd
# from streamlit.connections import ExperimentalBaseConnection
# import duckdb
#
# class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
#     def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
#         if 'database' in kwargs:
#             db = kwargs.pop('database')
#         else:
#             db = self._secrets['database']
#         return duckdb.connect(database=db, **kwargs)
#
#     def execute_query(self, query, params=None) -> pd.DataFrame:
#         """
#         Execute a SQL query and return the result as a pandas DataFrame.
#
#         Parameters:
#         query (str): The SQL query to be executed.
#         params (tuple or dict, optional): Parameters to be used in the query (if it's a parameterized query). Default is None.
#
#         Returns:
#         pd.DataFrame: The result of the query as a pandas DataFrame.
#         """
#         with self.connection.cursor() as cursor:
#             cursor.execute(query, params)
#             result = cursor.fetchall()
#             columns = [col[0] for col in cursor.description]
#             df = pd.DataFrame(result, columns=columns)
#             return df
#
#     def get_table_names(self) -> pd.DataFrame:
#         """
#         Get a list of all table names in the connected database.
#
#         Returns:
#         pd.DataFrame: A DataFrame containing the list of table names.
#         """
#         query = "SHOW TABLES;"
#         return self.execute_query(query)
#
#     def execute_update(self, query, params=None):
#         """
#         Execute an update/insert query and commit changes to the database.
#
#         Parameters:
#         query (str): The update/insert SQL query to be executed.
#         params (tuple or dict, optional): Parameters to be used in the query (if it's a parameterized query). Default is None.
#         """
#         with self.connection.cursor() as cursor:
#             cursor.execute(query, params)
#         self.connection.commit()


import pandas as pd
import streamlit as st
from streamlit.connections import ExperimentalBaseConnection
import duckdb

class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    def __init__(self, connection_name=None):
        super().__init__(connection_name)

    def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
        if 'database' in kwargs:
            db = kwargs.pop('database')
        else:
            db = self._secrets['database']
        return duckdb.connect(database=db, **kwargs)

    def execute_query(self, query, params=None) -> pd.DataFrame:
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            df = pd.DataFrame(result, columns=columns)
            return df

    def get_table_names(self) -> pd.DataFrame:
        query = "SHOW TABLES;"
        return self.execute_query(query)

    def execute_update(self, query, params=None):
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
        self.connection.commit()

# Initialize DuckDBConnection
connection = DuckDBConnection(connection_name="duckdb_connection")

# Main Streamlit app code
def main():
    st.title("DuckDB Connection Demo")

    # Connect to the database
    connection.connect(database="your_database_name")

    # User input section
    st.header("User Input")
    user_name = st.text_input("Enter your name:")
    user_email = st.text_input("Enter your email:")

    # Store user input in the database
    if st.button("Save"):
        query = f"INSERT INTO users (name, email) VALUES ('{user_name}', '{user_email}');"
        connection.execute_update(query)
        st.success("User data saved successfully!")

    # Example: Get a list of all table names in the connected database
    table_names_df = connection.get_table_names()

    # Display the table names
    st.write("List of Table Names:")
    st.write(table_names_df)

    # Example: Execute a SQL query and display the result as a DataFrame
    query = "SELECT * FROM users;"
    result_df = connection.execute_query(query)

    # Display the query result
    st.write("Query Result:")
    st.write(result_df)

if __name__ == "__main__":
    main()
