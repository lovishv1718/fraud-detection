# 🛡️ AI Fraud Detection System

A comprehensive machine learning-based fraud detection system for e-commerce transactions, featuring an ensemble model and an interactive Streamlit web dashboard.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Workflow](#project-workflow)
- [Contributors](#contributors)

## 🎯 Overview

This project implements an AI-powered fraud detection system designed to identify fraudulent e-commerce transactions in real-time. The system uses an ensemble of multiple machine learning models to achieve high accuracy and provides an intuitive web interface for both single transaction checks and batch analysis.

## ✨ Features

### 🔍 Core Capabilities
- **Real-time Fraud Detection**: Instant risk assessment for individual transactions
- **Batch Analysis**: Process thousands of transactions from CSV files
- **Interactive Dashboard**: Visual analytics with KPIs, charts, and transaction summaries
- **Ensemble Model**: Combines 5 different ML algorithms for robust predictions
- **Feature Engineering**: Automatic feature creation and preprocessing

### 🎨 Web Dashboard Features
- **Home Page**: Overview of system capabilities
- **Dashboard Tab**: 
  - Recent transaction summary
  - Key performance indicators (KPIs)
  - Fraud vs Safe transaction distribution charts
  - ROC curve visualization
- **Single Transaction Check**: Manual input form for instant fraud risk assessment
- **Batch Dataset Analysis**: Upload CSV files for bulk processing with downloadable results

## 📁 Project Structure

```
fraud-detection-college/
│
├── app.py                              # Streamlit web application
├── fraud_detection_ensemble.joblib     # Trained ensemble model
│
├── data_cleaning.ipynb                 # Data preprocessing and cleaning
├── data_visulization.ipynb             # Exploratory data analysis and visualizations
├── model_training.ipynb                 # Model training and ensemble creation
├── model_evaluation.ipynb              # Model performance evaluation
│
├── Fraudulent_E-Commerce_Transaction_Data.csv  # Original dataset
├── Cleaned_Fraud_Dataset.csv           # Preprocessed dataset
│
└── files/
    ├── demo.ipynb                      # Demo notebook
    └── sample_fraud_test.csv           # Sample test data
```

## 🛠️ Technologies Used

### Machine Learning
- **scikit-learn**: Logistic Regression, Random Forest, Voting Classifier
- **XGBoost**: Gradient Boosting Classifier
- **LightGBM**: Light Gradient Boosting Machine
- **CatBoost**: Categorical Boosting

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Visualization
- **matplotlib**: Static visualizations
- **seaborn**: Statistical data visualization

### Web Framework
- **streamlit**: Interactive web application
- **streamlit-lottie**: Animated UI components (optional)

### Model Persistence
- **joblib**: Model serialization and loading

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd "fraud detection college"
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost streamlit matplotlib seaborn joblib streamlit-lottie requests
```

Or create a `requirements.txt` file with:
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0
streamlit>=1.25.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
streamlit-lottie>=0.0.5
requests>=2.28.0
```

Then install:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**:
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically load the trained ensemble model

### Using the Dashboard

#### Single Transaction Check
1. Navigate to the **"Check Your Transaction"** tab
2. Fill in the transaction details:
   - Transaction Time (0-23 hours)
   - Customer Age
   - Transaction Amount
   - Merchant Category
   - Card Present (Yes/No)
   - Transaction Channel (Online/In-Store/Mobile)
   - Device Type (Desktop/Mobile/Tablet)
3. Click **"Analyze Transaction Risk"**
4. View the fraud probability and risk status

#### Batch Dataset Analysis
1. Navigate to the **"Check Your Transaction Dataset"** tab
2. Upload a CSV file containing transaction data
3. The system will automatically:
   - Clean and preprocess the data
   - Engineer features
   - Generate predictions for all transactions
4. View summary statistics, charts, and detailed results
5. Download the results with predictions as a CSV file

### Running the Notebooks

1. **Data Cleaning** (`data_cleaning.ipynb`):
   - Loads the original dataset
   - Removes unnecessary columns
   - Performs feature engineering
   - Exports cleaned dataset

2. **Data Visualization** (`data_visulization.ipynb`):
   - Creates visualizations for fraud patterns
   - Analyzes transaction distributions
   - Examines fraud by various features

3. **Model Training** (`model_training.ipynb`):
   - Trains individual models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
   - Creates ensemble model using Voting Classifier
   - Saves the trained model

4. **Model Evaluation** (`model_evaluation.ipynb`):
   - Loads the saved model
   - Evaluates performance metrics
   - Generates classification reports

## 📊 Model Performance

The ensemble model achieves the following performance metrics:

- **ROC-AUC Score**: 0.8542
- **PR-AUC Score**: 0.6038
- **Accuracy**: 90.90%
- **Precision (Fraud)**: 31.35%
- **Recall (Fraud)**: 68.55%
- **F1-Score (Fraud)**: 43.03%

### Model Architecture

The ensemble combines predictions from:
1. **Logistic Regression** (with class balancing)
2. **Random Forest** (200 estimators, class balanced)
3. **XGBoost** (with scale_pos_weight for imbalanced data)
4. **LightGBM** (optimized for speed and accuracy)
5. **CatBoost** (handles categorical features well)

Final predictions use **soft voting** (probability averaging) for optimal performance.

## 🔄 Project Workflow

1. **Data Collection**: Original dataset with 1.4M+ transactions
2. **Data Cleaning**: 
   - Remove irrelevant columns (IDs, addresses, IPs)
   - Extract temporal features from dates
   - Encode categorical variables (One-Hot and Frequency Encoding)
3. **Exploratory Data Analysis**: Visualize fraud patterns and distributions
4. **Model Training**: 
   - Split data (80% train, 20% test)
   - Train individual models with class balancing
   - Create ensemble model
5. **Model Evaluation**: Assess performance on test set
6. **Deployment**: Streamlit web application for real-time predictions



## 📝 Notes

- The model is trained on imbalanced data (~5% fraud rate)
- Class weights and scale_pos_weight are used to handle class imbalance
- The ensemble model file (`fraud_detection_ensemble.joblib`) must be present in the project directory for the app to function
- For best results, ensure uploaded CSV files match the expected feature schema

## 🔒 Use Cases

- **Banking**: Real-time transaction monitoring
- **E-commerce**: Fraud prevention for online purchases
- **Fintech**: Payment gateway security
- **Retail**: Point-of-sale fraud detection



**Note**: This is a machine learning project focused on fraud detection in e-commerce transactions. The model should be regularly retrained with new data to maintain accuracy.

