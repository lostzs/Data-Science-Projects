import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Check if the file exists
file_path = 'heart_attack.csv'
if not os.path.exists(file_path):
    st.error(f"The file '{file_path}' does not exist. Please check the file path.")
    st.stop()
else:
    # Load the dataset
    df = pd.read_csv(file_path)

# Function to calculate the probability of having CVD
def calculate_probability(user_values, filtered_df, variables):
    probability = 0.0

    for var, user_val in zip(variables, user_values):
        if var == 'gender':
            threshold_val = filtered_df[var].mode().iloc[0]
            probability += int(user_val == threshold_val)
        else:
            threshold_val = filtered_df[var].mean() + filtered_df[var].std()
            probability += int(user_val > threshold_val)

    probability /= len(variables)

    return probability

# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {'accuracy': accuracy, 'confusion_matrix': cm, 'predictions': y_pred}

    return results

# Streamlit app
st.title('Heart Disease Prediction App')

# Step 1: Basic Information
if 'user_values_basic' not in st.session_state:
    st.session_state.user_values_basic = {}
with st.form('basic_info_form'):
    st.header('Step 1: Enter Basic Information')
    variables_basic = ['age', 'gender', 'trestbps', 'cp', 'heart_disease']

    for var in variables_basic:
        if var == 'gender':
            st.session_state.user_values_basic[var] = st.text_input(f"Enter gender (M/F): ").upper()
        else:
            st.session_state.user_values_basic[var] = st.number_input(f"Enter value for {var}: ")

    if st.form_submit_button('Proceed to Step 2'):
        # Filter the dataset based on selected variables
        filtered_df_basic = df[variables_basic]

        # Train-test split for model evaluation
        X_basic = df.drop('heart_disease', axis=1)
        y_basic = df['heart_disease']
        X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X_basic, y_basic, test_size=0.2, random_state=42)

        # Evaluate models
        model_results_basic = evaluate_models(X_train_basic, X_test_basic, y_train_basic, y_test_basic)

        # Calculate the probability of having CVD
        initial_probability_basic = calculate_probability(list(st.session_state.user_values_basic.values()), filtered_df_basic, variables_basic)

        st.write(f"Initial Probability of having CVD: {initial_probability_basic:.2%}")

         # Step 2: Recommend Additional Information
        st.header('Step 2: Recommend Additional Information')
        if initial_probability_basic >= 0.35:
            additional_variables = ['chol', 'fbs', 'restecg', 'thalach', 'thal']
            st.write("Recommend getting additional information:")
        else:
            additional_variables = ['chol', 'fbs', 'restecg']
            st.write("Recommend getting the following information:")

        st.session_state.user_values_additional = {}  # Initialize user_values_additional

        for var in additional_variables:
            if var == 'gender':
                st.session_state.user_values_additional[var] = st.text_input(f"Enter gender (M/F): ").upper()
            else:
                st.session_state.user_values_additional[var] = st.number_input(f"Enter value for {var}: ")

# Step 3: Printable Ticket
if 'user_values_additional' in st.session_state:
    # Update the filtered dataset with additional information
    filtered_df_additional = df[variables_basic + list(st.session_state.user_values_additional.keys())]

    # Recalculate the probability with additional information
    updated_probability_additional = calculate_probability(
        list(st.session_state.user_values_basic.values()) + list(st.session_state.user_values_additional.values()), 
        filtered_df_additional,
        variables_basic + list(st.session_state.user_values_additional.keys())
    )

    # Display the updated probability percentage
    st.write(f"Updated Probability of having CVD: {updated_probability_additional:.2%}")

    # Calculate the difference in probability
    probability_change = updated_probability_additional - initial_probability_basic

    # Display the change in percentage
    st.write(f"Change in Probability: {probability_change:.2%}")

    # Recommendations
    st.write("\nRecommendations:")
    if probability_change >= 0.10:
        st.write("Recommend admission for further specialist checking. Mark as HIGH CHANCE OF CVD.")
    elif initial_probability_basic >= 0.35:
        st.write("Recommend admission to the ER for further investigation of symptoms. Mark as low probability for CVD. Check for other symptoms.")
    else:
        st.write("No specific recommendations at this time.")

    # Display model evaluation results for basic information
    st.write("\nModel Evaluation Results (Basic Information):")
    for model_name, result in model_results_basic.items():
        st.write(f"{model_name} Accuracy: {result['accuracy']:.2%}")
        st.write(f"{model_name} Confusion Matrix:\n{result['confusion_matrix']}\n")

        # Average the predictions across instances
        average_prediction = sum(result['predictions']) / len(result['predictions'])
        st.write(f"{model_name} Average Prediction (in percentage): {average_prediction:.2%}")

        # Visualization option report for basic information
        st.write("Visualization Option Report (Basic Information):")
        # Create subplots for confusion matrix and risk mapping
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion matrix
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'], ax=axes[0])
        axes[0].set_title(f'Confusion Matrix - {model_name}')

        # Risk mapping based on confusion matrix
        sensitivity = result['confusion_matrix'][1, 1] / (result['confusion_matrix'][1, 0] + result['confusion_matrix'][1, 1])
        specificity = result['confusion_matrix'][0, 0] / (result['confusion_matrix'][0, 0] + result['confusion_matrix'][0, 1])

        axes[1].set_title(f"Risk Mapping - {model_name}:")
        axes[1].text(0.5, 0.5, f"Sensitivity (True Positive Rate): {sensitivity:.2%}\n"
                               f"Specificity (True Negative Rate): {specificity:.2%}",
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1].axis('off')

        # Display the subplots
        st.pyplot(fig)

        # Map risk based on sensitivity and specificity
        if sensitivity > 0.70 and specificity > 0.70:
            st.write("Low Risk of CVD")
        elif sensitivity > 0.70 or specificity > 0.70:
            st.write("Moderate Risk of CVD")
        else:
            st.write("High Risk of CVD")

# ... (Previous code remains unchanged)

# Step 4: Printable Ticket
if 'user_values_additional' in st.session_state:
    # Update the filtered dataset with additional information
    filtered_df_additional = df[variables_basic + list(st.session_state.user_values_additional.keys())]

    # Recalculate the probability with additional information
    updated_probability_additional = calculate_probability(
        list(st.session_state.user_values_basic.values()) + list(st.session_state.user_values_additional.values()), 
        filtered_df_additional,
        variables_basic + list(st.session_state.user_values_additional.keys())
    )

    # Check if 'initial_probability_basic' is in session state, if not, initialize it
    if 'initial_probability_basic' not in st.session_state:
        st.session_state.initial_probability_basic = 0.0

    # Display the updated probability percentage
    st.write(f"Updated Probability of having CVD: {updated_probability_additional:.2%}")

    # Calculate the difference in probability
    probability_change = updated_probability_additional - st.session_state.initial_probability_basic

    # Display the change in percentage
    st.write(f"Change in Probability: {probability_change:.2%}")

    # Recommendations
    st.write("\nRecommendations:")
    if probability_change >= 0.10:
        st.write("Recommend admission for further specialist checking. Mark as HIGH CHANCE OF CVD.")
    elif st.session_state.initial_probability_basic >= 0.35:
        st.write("Recommend admission to the ER for further investigation of symptoms. Mark as high probability of CVD. Check for other symptoms.")
    else:
        st.write("No specific recommendations at this time.")

    # Display model evaluation results for basic information
    st.write("\nModel Evaluation Results (Basic Information):")
    for model_name, result in model_results_basic.items():
        st.write(f"{model_name} Accuracy: {result['accuracy']:.2%}")
        st.write(f"{model_name} Confusion Matrix:\n{result['confusion_matrix']}\n")

        # Average the predictions across instances
        average_prediction = sum(result['predictions']) / len(result['predictions'])
        st.write(f"{model_name} Average Prediction (in percentage): {average_prediction:.2%}")

        # Visualization option report for basic information
        st.write("Visualization Option Report (Basic Information):")
        # Create subplots for confusion matrix and risk mapping
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion matrix
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'], ax=axes[0])
        axes[0].set_title(f'Confusion Matrix - {model_name}')

        # Risk mapping based on confusion matrix
        sensitivity = result['confusion_matrix'][1, 1] / (result['confusion_matrix'][1, 0] + result['confusion_matrix'][1, 1])
        specificity = result['confusion_matrix'][0, 0] / (result['confusion_matrix'][0, 0] + result['confusion_matrix'][0, 1])

        axes[1].set_title(f"Risk Mapping - {model_name}:")
        axes[1].text(0.5, 0.5, f"Sensitivity (True Positive Rate): {sensitivity:.2%}\n"
                               f"Specificity (True Negative Rate): {specificity:.2%}",
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1].axis('off')

        # Display the subplots
        st.pyplot(fig)

        # Map risk based on sensitivity and specificity
        if sensitivity > 0.70 and specificity > 0.70:
            st.write("Low Risk of CVD")
        elif sensitivity > 0.70 or specificity > 0.70:
            st.write("Moderate Risk of CVD")
        else:
            st.write("High Risk of CVD")

    # Store updated initial_probability_basic in session state
    st.session_state.initial_probability_basic = updated_probability_additional

    # Display user inputs and additional information
    st.header('Step 4: Printable Ticket')
    
    # Include user inputs
    st.write("User Inputs:")
    for var, value in st.session_state.user_values_basic.items():
        st.write(f"{var}: {value}")

    st.write("Additional Information:")
    for var, value in st.session_state.user_values_additional.items():
        st.write(f"{var}: {value}")

    # Include results for each model (Basic Information)
    for model_name, result in model_results_basic.items():
        st.write(f"{model_name} Accuracy: {result['accuracy']:.2%}")
        st.write(f"{model_name} Confusion Matrix:\n{result['confusion_matrix']}\n")

        # Average the predictions across instances and display as percentage
        average_prediction = sum(result['predictions']) / len(result['predictions'])
        st.write(f"{model_name} Average Prediction (in percentage): {average_prediction:.2%}")

        # Recommendations (Basic Information)
        st.write("Recommendations (Basic Information):")
        probability_change = updated_probability_additional - st.session_state.initial_probability_basic
        if probability_change >= 0.10:
            st.write("Recommend admission for further specialist checking. Mark as HIGH CHANCE OF CVD.")
        elif st.session_state.initial_probability_basic >= 0.35:
            st.write("Recommend admission to the ER for further investigation of symptoms. Mark as high change of CVD.")
        else:
            st.write("No specific recommendations at this time.")

        # Display the updated probability percentage
        st.write(f"Updated Probability of having CVD: {updated_probability_additional:.2%}")

        # Map risk based on sensitivity and specificity
        sensitivity = result['confusion_matrix'][1, 1] / (result['confusion_matrix'][1, 0] + result['confusion_matrix'][1, 1])
        specificity = result['confusion_matrix'][0, 0] / (result['confusion_matrix'][0, 0] + result['confusion_matrix'][0, 1])

        st.write("Risk Mapping:")
        if sensitivity > 0.70 and specificity > 0.70:
            st.write("Risk of CVD: Low")
        elif sensitivity > 0.70 or specificity > 0.70:
            st.write("Risk of CVD: Moderate")
        else:
            st.write("Risk of CVD: High")
