import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import numpy as np

# Load the model and transformers
xgb_model = joblib.load('xgb_model_polynomial.pkl')
encoder = joblib.load('encoder.pkl')
poly = joblib.load('poly.pkl')

# Function to calculate PCA values based on trip distance and duration
def calculate_pca_values(trip_distance, trip_duration):
    pca1 = np.log1p(trip_distance) - np.log1p(trip_duration)
    pca2 = np.log1p(trip_duration) - np.log1p(trip_distance)
    return pca1, pca2

def predict_fare(trip_distance, trip_duration, tip_amount, is_holiday, 
                 pickup_time_of_day, pickup_season, passenger_count_category, 
                 pickup_day_type, pickup_location, dropoff_location):
    # Generalize zones into categories
    general_locations = {
        "downtown": ["Midtown Center", "Midtown East", "Lower Manhattan"],
        "suburbs": ["Manhattan Valley", "East Harlem South", "Upper West Side", "Upper East Side"],
        "airport": ["JFK Airport", "LaGuardia Airport"]
    }
    
    # Map general locations to zones (example mapping)
    if pickup_location in general_locations["downtown"]:
        puzone = "Midtown Center"
    elif pickup_location in general_locations["airport"]:
        puzone = "JFK Airport"
    else:
        puzone = "Manhattan Valley"
    
    if dropoff_location in general_locations["downtown"]:
        dozone = "Midtown Center"
    elif dropoff_location in general_locations["airport"]:
        dozone = "JFK Airport"
    else:
        dozone = "Manhattan Valley"
    
    # Calculate PCA values
    pca1, pca2 = calculate_pca_values(trip_distance, trip_duration)
    
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'trip_distance': [trip_distance],
        'trip_duration': [trip_duration],
        'tip_amount': [tip_amount],
        'PCA1': [pca1],
        'PCA2': [pca2],
        'is_holiday': [is_holiday],
        'pickup_time_of_day': [pickup_time_of_day],
        'pickup_season': [pickup_season],
        'passenger_count_category': [passenger_count_category],
        'pickup_day_type': [pickup_day_type],
        'PUzone': [puzone],
        'PUborough': ["Manhattan"],  # Assuming single borough for simplicity
        'DOzone': [dozone],
        'DOborough': ["Manhattan"]  # Assuming single borough for simplicity
    })
    
    # Encode the input data
    input_encoded = encoder.transform(input_data).toarray()
    
    # Apply polynomial transformations
    input_poly = poly.transform(input_encoded)
    
    # Predict the fare
    predicted_fare = xgb_model.predict(input_poly)
    
    # Ensure the predicted fare is non-negative
    predicted_fare = max(predicted_fare[0], 0)
    
    return predicted_fare

# Create the main window
root = tk.Tk()
root.title("Taxi Fare Predictor")
root.geometry("400x600")

# Create input fields
def create_label_and_entry(label_text, row):
    label = ttk.Label(root, text=label_text)
    label.grid(column=0, row=row, padx=10, pady=5, sticky=tk.W)
    entry = ttk.Entry(root, width=20)
    entry.grid(column=1, row=row, padx=10, pady=5)
    return entry

trip_distance_entry = create_label_and_entry("Trip Distance (km):", 0)
trip_duration_entry = create_label_and_entry("Trip Duration (minutes):", 1)
tip_amount_entry = create_label_and_entry("Tip Amount:", 2)
is_holiday_entry = create_label_and_entry("Holiday (1 for Yes, 0 for No):", 3)
pickup_time_of_day_entry = create_label_and_entry("Pickup Time of Day (morning/afternoon/evening/night):", 4)
pickup_season_entry = create_label_and_entry("Pickup Season (winter/spring/summer/autumn):", 5)
passenger_count_category_entry = create_label_and_entry("Passenger Count Category (low/medium/high):", 6)
pickup_day_type_entry = create_label_and_entry("Pickup Day Type (weekday/weekend):", 7)
pickup_location_entry = create_label_and_entry("Pickup Location (downtown/suburbs/airport):", 8)
dropoff_location_entry = create_label_and_entry("Dropoff Location (downtown/suburbs/airport):", 9)

# Label to display the result
result_label = ttk.Label(root, text="Predicted Fare: ")
result_label.grid(column=0, row=11, columnspan=2, padx=10, pady=20)

# Function to get input values and predict fare
def on_predict():
    trip_distance = float(trip_distance_entry.get())
    trip_duration = float(trip_duration_entry.get())
    tip_amount = float(tip_amount_entry.get())
    is_holiday = int(is_holiday_entry.get())
    pickup_time_of_day = pickup_time_of_day_entry.get()
    pickup_season = pickup_season_entry.get()
    passenger_count_category = passenger_count_category_entry.get()
    pickup_day_type = pickup_day_type_entry.get()
    pickup_location = pickup_location_entry.get()
    dropoff_location = dropoff_location_entry.get()
    
    fare = predict_fare(trip_distance, trip_duration, tip_amount, is_holiday, 
                        pickup_time_of_day, pickup_season, passenger_count_category, 
                        pickup_day_type, pickup_location, dropoff_location)
    
    result_label.config(text=f"Predicted Fare: ${fare:.2f}")

# Add a button to trigger the prediction
predict_button = ttk.Button(root, text="Predict Fare", command=on_predict)
predict_button.grid(column=0, row=10, columnspan=2, padx=10, pady=20)

# Run the application
root.mainloop()
