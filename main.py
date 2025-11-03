import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import io
import numpy as np 

# --- Lottie integration libraries and helper functions ---
try:
    from streamlit_lottie import st_lottie
    import requests

    LOTTIE_AVAILABLE = True

    def load_lottieurl(url):
        """Fetches Lottie animation data from a URL."""
        try:
            # Added a timeout for robustness against slow network requests
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()
            return None
        except requests.exceptions.RequestException:
            # Catches network errors (e.g., DNS failure, timeout)
            return None

    # Load Lottie animations from public URLs
    lottie_loading = load_lottieurl("https://lottie.host/80e98038-f9b1-4f10-90c7-573522f281e4/U09yD2g7iQ.json")
    lottie_success = load_lottieurl("https://lottie.host/a61c47a9-e85c-4286-9818-a6d10f855d49/w77e3L1l8n.json")

except ImportError:
    # If the libraries are not installed, set LOTTIE_AVAILABLE to False and skip animations
    st.warning("Animation library (streamlit-lottie) or requests library not found. Dashboard will run without animations.")
    LOTTIE_AVAILABLE = False
    lottie_loading = None
    lottie_success = None
    
# --- ML Model Wrapper Function (DITTO SAME LOGIC) ---
# This function encapsulates your core prediction logic for reuse in both single and batch modes.
def predict_fraud(data: pd.DataFrame, model_path="ensemble_model.pkl"):
    """
    Performs feature engineering and prediction using the loaded model.
    The logic here is kept DITTO SAME as the original request.
    """
    data_for_pred = data.copy()

    # DITTO SAME Feature Engineering
    data_for_pred['is_night'] = data_for_pred['transaction_time'].apply(lambda x: 1 if x <= 6 or x >= 22 else 0)
    # Handle potential division by zero for age (kept original logic which adds 1)
    data_for_pred['amount_per_age'] = data_for_pred['transaction_amount'] / (data_for_pred['customer_age'] + 1)
    data_for_pred['is_high_risk_merchant'] = data_for_pred['merchant_category'].apply(lambda x: 1 if x in [1, 3] else 0)

    # DITTO SAME Dropping columns if they exist in the input data
    # (Note: For manual input, these won't exist, but we check for robustness)
    for col in ['transaction_id', 'is_fraud', 'location']:
        if col in data_for_pred.columns:
            data_for_pred = data_for_pred.drop(col, axis=1)

    try:
        model = joblib.load(model_path) 
        predictions = model.predict(data_for_pred)
        probabilities = model.predict_proba(data_for_pred)[:, 1]
        
        # Create output DataFrame
        results = pd.DataFrame({
            'Prediction': predictions,
            'Fraud Probability': probabilities
        }, index=data.index)
        
        return results, None # Return results and no error
    except FileNotFoundError:
        return None, "🚫 Model file not found! Please ensure 'ensemble_model.pkl' is in the application directory."
    except Exception as e:
        return None, f"🚫 Error during prediction: {str(e)}"


# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the new 3-section layout and light theme
st.markdown("""
<style>
    /* Streamlit Main App Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* H1 Title styling */
    h1.fraud-title {
        color: #FF4B4B; /* Streamlit red */
        text-align: center;
        font-size: 4em;
        font-weight: 900;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* Wrapper for the three sections (tabs) with rounded borders */
    .stTabs {
        border: 2px solid #ccc; /* Light gray border */
        border-radius: 15px; /* Rounded edges */
        padding: 10px 20px 20px 20px; /* Padding inside the border */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        margin-top: 1.5rem;
    }
    /* Style for the active tab content area */
    .stTabs [data-testid="stVerticalBlock"] {
        padding-top: 1rem;
    }
    /* Metric Card Customization - Adjusted for Light Theme */
    [data-testid="stMetric"] {
        background-color: #f0f2f6; /* Very light gray/white background */
        padding: 20px;
        border-radius: 10px;
        color: #333333; /* Dark text for contrast */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px); /* Lift effect on hover */
    }
    /* Custom button styling for Manual Check */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #e63946;
    }
</style>
""", unsafe_allow_html=True)


# --- H1 TITLE ---
st.markdown('<h1 class="fraud-title">FRAUD</h1>', unsafe_allow_html=True)


# --- MAIN 3-SECTION LAYOUT (TABS) ---
tab1, tab2, tab3 = st.tabs([
    "📊 About Fraud & Dashboard", 
    "📝 Check Your Transaction", 
    "📂 Check Your Transaction Dataset"
])

# =========================================================================
# === SECTION 1: DASHBOARD (ABOUT FRAUD + RECENT TRANSACTIONS/GRAPHS) ===
# =========================================================================
with tab1:
    st.subheader("📊 About Fraud - Transaction Monitoring Dashboard")
    st.markdown("This dashboard provides an overview of predicted risk patterns in the latest analyzed dataset.")
    
    # --- MOCK DATA FOR DEMO ---
    if 'mock_df' not in st.session_state:
        # Create a small mock dataset to display "Recent Fraud" if no file is uploaded yet
        st.session_state.mock_df = pd.DataFrame({
            'transaction_id': [9001, 9002, 9003, 9004, 9005],
            'timestamp': ['14:30', '02:15', '11:00', '23:45', '09:00'],
            'amount': [1200.50, 89.99, 450.00, 3100.25, 55.00],
            'merchant_category': ['Electronics', 'Gas Station', 'Supermarket', 'Online Service', 'Restaurant'],
            'risk_status': ['✅ SAFE', '🚨 FRAUD', '✅ SAFE', '🚨 FRAUD', '✅ SAFE'],
            'fraud_prob': [0.12, 0.95, 0.05, 0.88, 0.02]
        })
    
    st.markdown("#### 🔍 Recent Transaction Summary")
    
    # Use the first 5 records of the last uploaded data or mock data
    display_df = st.session_state.get('last_analyzed_df', st.session_state.mock_df)
    
    # Apply a simplified styling for the recent transactions
    def style_recent_rows(row):
        return ['background-color: #ffcccc; color: black'] * len(row) if '🚨 FRAUD' in row.get('Risk Status', row.get('risk_status')) else [''] * len(row)
    
    st.dataframe(
        display_df.head(5).style.apply(style_recent_rows, axis=1),
        use_container_width=True
    )
    
    # --- Display Graphs (reusing logic from Section 3 if data exists) ---
    if 'full_df' in st.session_state:
        full_df = st.session_state.full_df
        
        st.markdown("#### 📈 Dashboard Visualizations (Based on Last Upload)")
        
        total_transactions = len(full_df)
        fraud_predicted = full_df["Prediction"].sum()
        fraud_rate = (fraud_predicted / total_transactions) * 100 if total_transactions > 0 else 0
        safe_transactions = total_transactions - fraud_predicted
        
        col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)

        with col_kpi_1:
            st.metric(label="Total Analyzed", value=f"{total_transactions:,}")
        with col_kpi_2:
            st.markdown(f'<div style="border: 2px solid #FF4B4B; padding: 10px; border-radius: 8px; background-color: #ffeaea;">'
                        f'  <p style="margin: 0; font-size: 14px; color: #FF4B4B;">Predicted Fraud Cases</p>'
                        f'  <h3 style="margin: 0; color: #FF4B4B; font-size: 24px;">{fraud_predicted:,}</h3>'
                        f'  <p style="margin: 0; font-size: 10px; color: #FF4B4B;">High Risk</p>'
                        f'</div>', unsafe_allow_html=True)
        with col_kpi_3:
            st.metric(label="Predicted Safe Cases", value=f"{safe_transactions:,}")
        with col_kpi_4:
            st.metric(label="Predicted Fraud Rate", value=f"{fraud_rate:.2f} %", delta_color="off")
        
        # Display the prediction distribution chart
        st.markdown("---")
        st.markdown("#### Distribution Chart")
        pie_data = full_df["Prediction"].value_counts().rename({0: 'Safe (0)', 1: 'Fraud (1)'})
        
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ["#00cc96", "#FF4B4B"]
        
        ax.pie(
            pie_data, 
            labels=pie_data.index, 
            autopct=lambda p: '{:.1f}%\n({:,.0f})'.format(p, (p/100)*pie_data.sum()), 
            startangle=90, 
            colors=colors,
            wedgeprops=dict(width=0.4),
            pctdistance=0.8,
            textprops={'fontsize': 12, 'color': 'black'}
        )
        ax.set_title("Fraud vs. Safe Transactions Count", fontsize=14)
        ax.axis("equal")
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        st.pyplot(fig)


# =========================================================================
# === SECTION 2: MANUAL CHECK (SINGLE TRANSACTION INPUT) ===
# =========================================================================
with tab2:
    st.subheader("📝 Check Your Transaction")
    st.markdown("Enter the details of a single transaction to get an instant fraud risk assessment.")
    
    with st.form("single_transaction_form"):
        st.markdown("##### Transaction Details")
        
        col_form_1, col_form_2 = st.columns(2)
        
        with col_form_1:
            transaction_time = st.slider(
                "Transaction Time (Hour of Day 0-23)", 
                min_value=0, max_value=23, value=12, step=1,
                help="The hour of the transaction (e.g., 12 for 12:00 PM)."
            )
            customer_age = st.number_input(
                "Customer Age (Years)", 
                min_value=18, max_value=100, value=30, step=1
            )
        
        with col_form_2:
            transaction_amount = st.number_input(
                "Transaction Amount (Monetary Value)", 
                min_value=0.01, value=500.00, step=10.00, format="%.2f"
            )
            merchant_category = st.selectbox(
                "Merchant Category (ID)",
                options=[1, 2, 3, 4, 5, 6, 7],
                help="Categorical ID of the merchant. (Your model expects integer inputs)."
            )
        
        submit_button = st.form_submit_button("Analyze Transaction Risk")

    if submit_button:
        # Create a DataFrame for prediction (matching the expected format)
        input_data = pd.DataFrame({
            'transaction_time': [transaction_time],
            'customer_age': [customer_age],
            'transaction_amount': [transaction_amount],
            'merchant_category': [merchant_category],
        })
        
        # Run prediction
        results, error = predict_fraud(input_data)
        
        if error:
            st.error(error)
        elif results is not None:
            prediction = results.iloc[0]['Prediction']
            probability = results.iloc[0]['Fraud Probability']
            
            st.markdown("---")
            st.markdown("#### 🎯 Prediction Result")
            
            if prediction == 1:
                st.error(f"🚨 **HIGH RISK: FRAUD DETECTED**")
                st.markdown(f"**Fraud Probability:** <span style='font-size: 24px; color: #FF4B4B;'>**{probability:.4f}**</span>", unsafe_allow_html=True)
                if LOTTIE_AVAILABLE and lottie_success:
                     st_lottie(lottie_success, height=100, key="single_fraud_anim", speed=1)
            else:
                st.success(f"✅ **LOW RISK: TRANSACTION IS SAFE**")
                st.markdown(f"**Fraud Probability:** <span style='font-size: 24px; color: #00cc96;'>**{probability:.4f}**</span>", unsafe_allow_html=True)
            
            st.info(f"The model predicted the transaction is: **{'Fraud' if prediction == 1 else 'Safe'}**.")


# =========================================================================
# === SECTION 3: BATCH CHECK (CSV UPLOAD) ===
# =========================================================================
with tab3:
    st.subheader("📂 Check Your Transaction Dataset")
    st.markdown("Upload a large CSV file to analyze a batch of transactions and view aggregate results.")

    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file here",
        type=["csv"],
        key="batch_uploader",
        help="The CSV must contain features expected by the model (transaction_time, transaction_amount, customer_age, merchant_category)."
    )

    if uploaded_file is None:
        st.info("👈 Please upload a CSV file to begin batch analysis.")
    else:
        # --- PROCESSING WITH LOADING ANIMATION ---
        with st.spinner('Model is processing and analyzing transactions...'):
            if LOTTIE_AVAILABLE and lottie_loading:
                col_lottie_load, col_lottie_text = st.columns([1, 4])
                with col_lottie_load:
                    st_lottie(lottie_loading, height=100, key="batch_loading_anim")
                with col_lottie_text:
                    st.write("Hang tight! Your data is being cleaned, features are being engineered, and the ensemble model is making predictions.")
            
            # Load and prepare data
            try:
                data = pd.read_csv(uploaded_file)
                
                # --- DITTO SAME MODEL LOGIC EXECUTION ---
                results_df, error = predict_fraud(data)
                
                if error:
                    st.error(error)
                    st.session_state.last_analyzed_df = pd.DataFrame()
                    st.session_state.full_df = pd.DataFrame()
                else:
                    # Prepare final output dataframe
                    full_df = data.copy()
                    full_df['Prediction'] = results_df['Prediction']
                    full_df['Fraud Probability'] = results_df['Fraud Probability']
                    full_df['Risk Status'] = full_df['Prediction'].apply(lambda x: '🚨 FRAUD' if x == 1 else '✅ SAFE')
                    
                    st.session_state.full_df = full_df
                    # Update the dashboard view data
                    st.session_state.last_analyzed_df = full_df.copy().drop(columns=['Prediction', 'Fraud Probability'], errors='ignore')
                    
                    st.success("✅ Batch prediction model executed successfully!")
                    if LOTTIE_AVAILABLE and lottie_success:
                         st_lottie(lottie_success, height=150, key="batch_success_anim", speed=1)

                    # --- Display Results ---
                    st.markdown("---")
                    st.subheader("📝 Analysis Summary")

                    total_transactions = len(full_df)
                    fraud_predicted = full_df["Prediction"].sum()
                    safe_transactions = total_transactions - fraud_predicted
                    fraud_rate = (fraud_predicted / total_transactions) * 100 if total_transactions > 0 else 0

                    col_summary_1, col_summary_2 = st.columns(2)
                    with col_summary_1:
                        st.metric(label="Total Transactions Analyzed", value=f"{total_transactions:,}")
                    with col_summary_2:
                        st.metric(label="Total Predicted Fraud Cases", value=f"{fraud_predicted:,}", delta=f"{fraud_rate:.2f}% of total", delta_color="inverse")
                    
                    # --- DETAILED DATA DISPLAY & DOWNLOAD ---
                    
                    st.markdown("---")
                    st.subheader("📈 Prediction Distribution and Data Details")

                    # Use columns to display chart and download button side-by-side
                    col_chart, col_data = st.columns([1, 1])

                    with col_chart:
                        st.markdown("#### Prediction Class Distribution")
                        # Prediction Distribution Pie Chart (Kept DITTO SAME)
                        pie_data = full_df["Prediction"].value_counts().rename({0: 'Safe (0)', 1: 'Fraud (1)'})
                        
                        fig, ax = plt.subplots(figsize=(5, 5))
                        colors = ["#00cc96", "#FF4B4B"]
                        
                        ax.pie(
                            pie_data, 
                            labels=pie_data.index, 
                            autopct=lambda p: '{:.1f}%\n({:,.0f})'.format(p, (p/100)*pie_data.sum()), 
                            startangle=90, 
                            colors=colors,
                            wedgeprops=dict(width=0.4),
                            pctdistance=0.8,
                            textprops={'fontsize': 10, 'color': 'black'}
                        )
                        ax.axis("equal")
                        centre_circle = plt.Circle((0,0),0.70,fc='white')
                        fig.gca().add_artist(centre_circle)
                        st.pyplot(fig)
                    
                    with col_data:
                        st.markdown("#### Download Results")
                        st.markdown("The table below shows the full dataset with added **Prediction** and **Fraud Probability** columns.")
                        # Download Button
                        csv = full_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Prediction Results (CSV)",
                            data=csv,
                            file_name="fraud_predictions_analyzed.csv",
                            mime="text/csv",
                            key='batch_download_csv',
                            help="Download the original data file with added Prediction and Fraud Probability columns."
                        )
                        st.markdown("---")
                        
                        # ROC Curve (Reusing logic for visual confirmation)
                        st.markdown("#### ROC Curve")
                        try:
                            y_true = np.where(full_df["Prediction"] == 1, 1, 0)
                            fpr, tpr, _ = roc_curve(y_true, full_df["Fraud Probability"])
                            roc_auc = auc(fpr, tpr)
                        except ValueError:
                             fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5 

                        fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
                        ax_roc.plot(fpr, tpr, color='#00cc96', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
                        ax_roc.plot([0, 1], [0, 1], color='#FF4B4B', linestyle='--', label='Random Guess')
                        ax_roc.set_xlabel("False Positive Rate", fontsize=10)
                        ax_roc.set_ylabel("True Positive Rate", fontsize=10)
                        ax_roc.legend(loc="lower right")
                        ax_roc.grid(True, linestyle=':', alpha=0.6)
                        st.pyplot(fig_roc)

                    # --- Display Dataframes using expanders ---
                    st.markdown("---")
                    st.markdown("##### Detailed Transaction View")
                    def highlight_fraud_rows(row):
                        """Applies a style to the whole row if 'Risk Status' is '🚨 FRAUD'."""
                        if row['Risk Status'] == '🚨 FRAUD':
                            return ['background-color: #ffcccc; color: black; font-weight: bold'] * len(row)
                        return [''] * len(row)

                    with st.expander("Show Detailed Results (Fraud Highlighted)", expanded=False):
                        styled_df = full_df.style.apply(highlight_fraud_rows, axis=1)
                        st.dataframe(styled_df, use_container_width=True)


            except Exception as e:
                st.error(f"🚫 An unexpected error occurred during processing: {str(e)}")
                st.warning("Please check that your uploaded CSV file has the necessary columns and valid numeric data.")
