
# Advanced Data Analytics for NYC Taxi Trips

## Project Overview
This project analyzes the New York City taxi trip dataset for the year 2023, employing Python for comprehensive data preprocessing, exploratory analysis, and predictive modeling with machine learning techniques. The goal is to understand key factors influencing taxi trip pricing and other related outcomes. The dataset includes over 30 million records, detailing pickup and dropoff times, locations, distances traveled, fares, and more.

## Dataset
The data is sourced from the NYC Open Data portal and has been pre-processed into four partitions to manage the large volume. Each file contains detailed trip records including:
- VendorID
- tpep_pickup_datetime
- tpep_dropoff_datetime
- passenger_count
- trip_distance
- RatecodeID
- store_and_fwd_flag
- PULocationID
- DOLocationID
- payment_type
- fare_amount
- extra
- mta_tax
- tip_amount
- tolls_amount
- improvement_surcharge
- total_amount
- congestion_surcharge

## Installation
Clone this repository to your local machine using the following command:
```
git clone https://github.com/doli99/Advanced_Data_Analytics_UNIL_MsCM_BA_Mariam_Dolidze.git
```

## Requirements
This project uses Python 3. Install the required packages with:
```
pip install -r requirements.txt
```
Required Packages: 
!pip install pyarrow
!pip install fastparquet
!pip install geopandas


## Usage
Each Jupyter Notebook in the repository corresponds to different stages of the project:
- **1.** Data Fetching From Official Website And Combination For Further Analysis
  - Using this notebook we will fetch 2023 parquet files from NYC website and combine into combined dataframe
  - For memory optimization this step does not have to be repeated as we have prvided saved datset in our data storage
- **2.** Initial Data Cleaning from nulls, negatives and duplicates
  - In this notebook we load combined dataset for preliminary cleaning
  - Get rid of null values, duplicates and uneccesary features for our future analysis
  - We save the results in cleaned version thus negating the need to run this notebook for modeling as the dataset is quite large 
- **3.** Feature Enginerring and Stratified Sampling
  - In this dataset we create features for our dataset based on domain knowledge and litarature review
  - We create sampled dataset for further EDA and modeling as the initial dataset is too large
  - Thus to validate the notebook one can load the feature engineered dataset and test that it works    
- **4.** EDA on Stratified Sampled Data
  - For eda we use feature engineered and sample ddataset for run time and memory optimiziation
  - We use univariate, bivariate analysis to understand relationships and data structures for future modeling
- **5.** Data Preparation for Modeling and Model building and Evaluation
  - In this notebook we normalize data, scale and encode it and drop unecessary features
  - we use the new modeling dataset for model building and testing
  - This notebook can be initialised from model testing part where one can use already created modeling parquet file
- **6.** GUI (NOTEBOOK VERSION)
  - In this notebook we created interactive GUI using tkinter for visualising the sample dataset easier for users.
  - This file in notebook can be run in anaconda environment using all the imports and requirements listed in the requirements.txt


## Contact
**Mariam Dolidze**  
Email: [mariam.dolidze@unil.ch](mailto:mariam.dolidze@unil.ch)  
Project Link: [Advanced Data Analytics UNIL MsCM BA Mariam Dolidze](https://github.com/doli99/Advanced_Data_Analytics_UNIL_MsCM_BA_Mariam_Dolidze)
