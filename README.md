# Medical Insurance Premium Prediction Project

## Project Overview
The **Medical Insurance Premium Prediction** project aims to build a machine learning model that predicts the annual premium cost for customers based on their health-related parameters. By leveraging this model, we can help individuals and insurance companies make informed decisions about medical coverage and its pricing.

## Problem Context
The rising cost of healthcare services has made it essential for individuals to plan their medical expenses carefully. Understanding the factors influencing insurance premiums can:
- Enable customers to budget for their healthcare.
- Help insurance companies design personalized insurance plans.
- Provide insights into health-related risks and their impact on premium costs.

This project uses a dataset of health parameters provided voluntarily by nearly 1,000 individuals to create a predictive model for insurance premiums. The premium prices are represented in INR (₹) for yearly coverage.

## Dataset Description
The dataset includes the following attributes:  
- **Age**: The age of the individual (in years).  
- **Diabetes**: Whether the individual has diabetes (Yes/No).  
- **BloodPressureProblems**: Whether the individual has a history of blood pressure issues (Yes/No).  
- **AnyTransplants**: Whether the individual has undergone any organ transplants (Yes/No).  
- **AnyChronicDiseases**: Whether the individual has chronic diseases (Yes/No).  
- **Height**: The height of the individual (in cm).  
- **Weight**: The weight of the individual (in kg).  
- **KnownAllergies**: Whether the individual has known allergies (Yes/No).  
- **HistoryOfCancerInFamily**: Whether there is a family history of cancer (Yes/No).  
- **NumberOfMajorSurgeries**: The number of major surgeries the individual has undergone.  
- **PremiumPrice (Target Variable)**: The annual medical insurance premium (in INR ₹).

## Goals
1. **Exploratory Data Analysis (EDA)**:  
   - Analyze and visualize data to uncover patterns and relationships.
   - Identify key features that influence premium costs.
   - Handle missing values and detect outliers.
   
2. **Model Development**:  
   - Build and evaluate regression models to predict the premium cost.
   - Compare different algorithms to determine the best-performing model.
   - Optimize the model to improve accuracy and generalization.

3. **Insights and Impact**:  
   - Highlight the most significant factors influencing premiums.
   - Provide actionable insights for both customers and insurance providers.


## Run predict.py

To run this code from the console, follow these steps:

1. Ensure you have Python installed on your system. This code requires Python 3.x.
2. Ensure you have the `ridge_model.bin` file in the same directory as `predict.py`. This file should contain the serialized model and test data.
3. Open a terminal or command prompt and navigate to the directory where `app.py` is located.
4. Run the application by executing the following command:
   ```
   python predict.py
   ```
6. The Flask application will start, and you should see output indicating that the server is running on `http://0.0.0.0:9696`.
7. You can now send POST requests to `http://0.0.0.0:9696/predict` with JSON data containing customer information to get price predictions.


## Notebook: `predict_test.ipynb`

The `predict_test.ipynb` notebook is designed to demonstrate the prediction capabilities of the trained machine learning model on new, unseen data. This notebook includes the following sections:

1. **Loading the Model**:
   - Load the pre-trained model from the saved file.
   - Ensure all necessary dependencies and libraries are imported.

2. **Preparing Test Data**:
   - Load the test dataset containing new individuals' health parameters.
   - Perform any required preprocessing steps to match the training data format.

3. **Making Predictions**:
   - Use the loaded model to predict the insurance premiums for the test dataset.
   - Store and display the predicted premium values alongside the test data.

This notebook serves as a practical guide for applying the developed model to real-world data, ensuring that the predictions are accurate and reliable.

# To build and run the Docker image for the Medical Insurance Prediction project, follow these steps:

1. **Build the Docker Image**:
   - Navigate to the project directory where the Dockerfile is located.
   - Run the following command to build the Docker image and tag it as `insurance`:
     ```sh
     docker build -t insurance .
     ```

2. **Run the Docker Container**:
   - After building the image, run the container interactively and remove it after it exits:
     ```sh
     docker run -it --rm insurance
     ```
This will start the server and then you can make post request to it. 

3. **Execute the Jupyter Notebook**:
   - Navigate to the notebook named `predict_test.ipynb`.
   - Open the notebook and run all cells to perform the predictions.