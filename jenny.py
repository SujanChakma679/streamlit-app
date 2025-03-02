import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic']

st.set_page_config(layout="wide", page_title="Advanced Data Analysis Dashboard")

st.title("Advanced Data Analysis Dashboard")

# Sidebar for uploads and settings
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding="shift_jis")
        except UnicodeDecodeError:
            st.error("Unable to determine file encoding. Please save the file in UTF-8 or Shift-JIS format.")
            st.stop()
    
    st.write("### Data Preview")
    st.write(df.head())
    
    # Convert non-numeric columns that should be categorical
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except ValueError:
            df[col] = df[col].astype('category')
    
    # Data Filtering
    st.sidebar.header("Filter Data")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    filtered_df = df.copy()
    for col in numeric_columns:
        min_val, max_val = df[col].min(), df[col].max()
        if not np.isnan(min_val) and not np.isnan(max_val):
            filter_range = st.sidebar.slider(f"Filter {col}", min_value=float(min_val), max_value=float(max_val), 
                                            value=(float(min_val), float(max_val)))
            filtered_df = filtered_df[(filtered_df[col] >= filter_range[0]) & (filtered_df[col] <= filter_range[1])]
    
    for col in categorical_columns:
        unique_values = df[col].dropna().unique()
        selected_values = st.sidebar.multiselect(f"Filter {col}", options=unique_values, default=unique_values)
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    st.write("### Filtered Data Preview")
    st.write(filtered_df.head())
    
    # Basic statistical summary
    st.write("### Basic Statistics")
    st.write(filtered_df.describe())
    
    # Ensure numeric columns are not empty before transformations
    if numeric_columns:
        # Handling missing values for clustering and PCA
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(filtered_df[numeric_columns])
        filtered_df_imputed = pd.DataFrame(imputed_data, columns=numeric_columns, index=filtered_df.index)
        
        # Standardizing Data before PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(filtered_df_imputed)
    else:
        st.warning("No numeric columns available for PCA or clustering.")
        scaled_data = None
        filtered_df_imputed = None
    
    # KMeans Clustering
    if numeric_columns and not filtered_df.empty and scaled_data is not None:
        st.sidebar.header("Clustering")
        cluster_column = st.sidebar.selectbox("Select column for clustering", numeric_columns)
        n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        
        if len(filtered_df) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            filtered_df['Cluster'] = kmeans.fit_predict(scaled_data)
            
            fig_cluster = px.scatter(filtered_df, x=numeric_columns[0], y=numeric_columns[1], color=filtered_df['Cluster'].astype(str),
                                     title="KMeans Clustering Visualization")
            st.plotly_chart(fig_cluster)
        else:
            st.warning("Not enough data points for clustering. Reduce the number of clusters or provide more data.")
    
    # PCA for Dimensionality Reduction
    if len(numeric_columns) > 2 and not filtered_df.empty and scaled_data is not None:
        st.sidebar.header("PCA Analysis")
        n_components = st.sidebar.slider("Select number of PCA components", min_value=2, max_value=min(5, len(numeric_columns)), value=2)
        
        pca = PCA(n_components=n_components)
        pca_results = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)], index=filtered_df.index)
        if 'Cluster' in filtered_df:
            pca_df['Cluster'] = filtered_df['Cluster']
        
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color=pca_df['Cluster'].astype(str) if 'Cluster' in pca_df else None,
                              title="PCA Projection")
        st.plotly_chart(fig_pca)
    
    # Download options
    st.sidebar.header("Download Results")
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(label="Download Filtered Data (CSV)", data=csv, file_name="filtered_data.csv", mime="text/csv")
    
    st.markdown("---")
    st.write("Explore your data with advanced interactive visualizations and machine learning tools. Customize plots, perform clustering, and apply PCA using the sidebar!")
else:
    st.write("Please upload a CSV file.")
