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

## Importance
This project is a step toward solving a crucial financial problem by empowering individuals and businesses with predictive analytics. Understanding premium costs will enable people to make better financial decisions and manage healthcare expenses effectively.

---

## Tools and Techniques
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Machine Learning Models**: Linear Regression, Decision Tree Regressor, Random Forest, Gradient Boosting Models.
- **Performance Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² score.
