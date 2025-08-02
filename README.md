# calories_burnt_prediction

This project focuses on predicting the number of calories burnt during a workout session based on various physical and activity-related features. It utilizes several machine learning models to achieve this prediction.

# Project Structure

The project is contained within a single Jupyter Notebook (`Calories_Burnt_Prediction_using_Machine_Learning.ipynb`).

# Dependencies

The project uses the following Python libraries:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `sklearn` (for model selection, preprocessing, metrics, and various models like `SVC`, `LinearRegression`, `Lasso`, `Ridge`, `RandomForestRegressor`, `StandardScaler`, `LabelEncoder`, `mean_absolute_error`)
-   `xgboost` (for `XGBRegressor`)
-   `warnings`


# Dataset
The dataset used for this project is loaded from a CSV file named calories.csv. The dataset is expected to contain the following columns:

User_ID: Unique identifier for each user.
Gender: Gender of the user (male/female).
Age: Age of the user.
Height: Height of the user.
Weight: Weight of the user.
Duration: Duration of the workout.
Heart_Rate: Heart rate during the workout.
Body_Temp: Body temperature after the workout.
Calories: Calories burnt (target variable).

# Project Workflow
The Jupyter Notebook performs the following steps:

Import Libraries: Imports all necessary Python libraries.

Load Data: Reads the calories.csv file into a pandas DataFrame.

Exploratory Data Analysis (EDA):
Displays the first few rows of the DataFrame (df.head()).
Checks the shape of the DataFrame (df.shape).
Provides information about the DataFrame, including data types and non-null counts (df.info()).
Generates descriptive statistics of the numerical columns (df.describe()).
Visualizes the relationship between 'Height' and 'Weight' using a scatter plot.
Visualizes the relationship between 'Calories' and selected features ('Age', 'Height', 'Weight', 'Duration') using scatter plots.
Visualizes the distribution of float features using distribution plots.

Data Preprocessing:

Encode Gender: Converts 'Gender' from categorical (male/female) to numerical (0/1) representation.
Feature Selection: Drops 'User_ID', 'Weight', and 'Duration' columns as they might be redundant or less impactful for prediction based on initial analysis (or high correlation with other features).
Train-Validation Split: Splits the dataset into training and validation sets (90% for training, 10% for validation).
Feature Scaling: Normalizes the features using StandardScaler to ensure stable and fast training of machine learning models.

Model Training and Evaluation:

Initializes a list of several regression models: LinearRegression, XGBRegressor, Lasso, RandomForestRegressor, and Ridge.

Iterates through each model:

Trains the model on the scaled training data (X_train, Y_train).
Makes predictions on both the training and validation sets.
Calculates and prints the Mean Absolute Error (MAE) for both training and validation predictions to assess model performance.
