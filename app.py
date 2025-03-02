import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots

plt.rcParams['font.family'] = 'Meiryo'

# Set page configuration for a wide layout
st.set_page_config(layout="wide", page_title="Data Visualization and Analysis Dashboard")

# App title
st.title("Data Visualization and Analysis Dashboard")

# Create tabs for different visualization features
tab1, tab2, tab3, tab4 = st.tabs(["Normal Distribution", "Density Plot", "Pair Plot", "Joint Plot"])

# Tab 1: Normal Distribution Experiment
with tab1:
    st.subheader('確率分布の実験')

    st.markdown("### 正規分布\n\n母数（パラメータ）を変化させたときのグラフの変化の確認")

    # Sidebar for Normal Distribution controls (only shown in this tab)
    st.sidebar.header("Distribution Settings")

    # Sliders for mean (mu) and variance (sigma)
    mu = st.sidebar.slider('正規分布の期待値', min_value=-5.0, max_value=5.0, step=0.01, value=0.0, key='norm_mu')
    sigma = st.sidebar.slider('正規分布の分散', min_value=0.1, max_value=20.0, step=0.1, value=1.0, key='norm_sigma')

    # Additional sidebar options for customization
    show_cdf = st.sidebar.checkbox("Show Cumulative Distribution Function (CDF)", value=False, key='norm_cdf')
    show_samples = st.sidebar.checkbox("Show Random Samples", value=False, key='norm_samples')
    num_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=1000, step=10, value=100, key='norm_samples_num')
    grid_on = st.sidebar.checkbox("Show Grid", value=True, key='norm_grid')
    line_style = st.sidebar.selectbox("Line Style", options=['-', '--', '-.', ':'], index=0, key='norm_line_style')

    # Generate data for PDF and CDF
    x_1 = np.linspace(-10, 10, 100)
    pdf = stats.norm.pdf(x_1, mu, np.sqrt(sigma))
    cdf = stats.norm.cdf(x_1, mu, np.sqrt(sigma))

    # Create figure and axis for plotting
    fig_norm, ax1 = plt.subplots(figsize=(10, 6))

    # Plot PDF
    ax1.plot(x_1, pdf, label='Probability Density Function (PDF)', linestyle=line_style, color='blue')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Density')
    ax1.set_title('Normal Distribution (PDF)')
    ax1.legend()

    # Optionally plot CDF
    if show_cdf:
        ax2 = ax1.twinx()
        ax2.plot(x_1, cdf, label='Cumulative Distribution Function (CDF)', linestyle=line_style, color='red')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend(loc='upper left')

    # Optionally plot random samples
    if show_samples:
        samples = stats.norm.rvs(loc=mu, scale=np.sqrt(sigma), size=num_samples)
        ax1.scatter(samples, np.zeros_like(samples), color='green', alpha=0.5, label=f'Samples (n={num_samples})')
        ax1.legend()

    # Toggle grid
    ax1.grid(grid_on)

    # Display statistical properties
    st.subheader("Statistical Properties")
    stats_desc = {
        "Mean (μ)": mu,
        "Variance (σ²)": sigma,
        "Standard Deviation (σ)": np.sqrt(sigma),
        "Skewness": 0,  # Normal distribution has zero skewness
        "Kurtosis": 0,  # Normal distribution has zero excess kurtosis
    }
    st.write(pd.DataFrame([stats_desc]))

    # Display the plot in Streamlit
    st.pyplot(fig_norm)

    # Footer
    st.markdown("---")
    st.write("Adjust the parameters in the sidebar to explore the normal distribution. Use the checkboxes to toggle additional features like CDF and random samples.")

# Tab 2: Density Plot Generator
with tab2:
    # Streamlit app title
    st.title("Density Plot Generator")

    # Sidebar for Density Plot controls (only shown in this tab)
    st.sidebar.header("Density Plot Settings")

    # Normal Distribution Settings
    std_dev = st.sidebar.slider("Standard Deviation", min_value=0.1, max_value=20.0, value=1.0, key='density_std_dev')
    show_cdf = st.sidebar.checkbox("Show Cumulative Distribution Function (CDF)", key='density_cdf')
    show_samples = st.sidebar.checkbox("Show Random Samples", key='density_samples')
    n_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=1000, value=100, key='density_samples_num')
    show_grid = st.sidebar.checkbox("Show Grid", value=True, key='density_grid')
    line_style = st.sidebar.selectbox("Line Style", ["-", "--", "-.", ":"], key='density_line_style')

    # Option to use random data instead of uploading a CSV
    use_random_data = st.checkbox("Use random data instead of uploading a file", key='density_random_data')

    if use_random_data:
        # Generate random data
        st.subheader("Generate Random Data")
        data_size = st.slider("Select number of data points", min_value=50, max_value=1000, value=200, key='density_data_size')
        mean_value = st.number_input("Mean", value=0.0, key='density_mean')
        data = np.random.normal(loc=mean_value, scale=std_dev, size=data_size)
        
        # Plot density plot
        st.subheader("Density Plot")
        fig, ax = plt.subplots()
        sns.kdeplot(data, fill=True, linestyle=line_style, ax=ax)
        
        if show_cdf:
            sns.ecdfplot(data, ax=ax)
        
        if show_grid:
            ax.grid(True)
        
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title("Density Plot of Generated Data")
        st.pyplot(fig)
        
        if show_samples:
            st.subheader("Random Samples")
            st.write(np.random.choice(data, n_samples))
    else:
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key='density_file_uploader')

        if uploaded_file is not None:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)

            # Preprocessing: Handle missing values and convert non-numeric data
            st.subheader("Data Preprocessing")
            st.write("Checking for missing values and converting data...")
            
            # Fill missing values with the column mean (for numerical columns)
            for col in df.select_dtypes(include=['number']).columns:
                df[col] = df[col].fillna(df[col].mean())
            
            # Convert non-numeric columns if needed
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    df = df.drop(columns=[col])
            
            # Select a column for density plot
            numerical_columns = df.select_dtypes(include=['number']).columns
            if len(numerical_columns) > 0:
                column = st.selectbox("Select a numerical column", numerical_columns, key='density_column')

                # Plot density plot
                st.subheader(f"Density Plot for {column}")
                fig, ax = plt.subplots()
                sns.kdeplot(df[column], fill=True, linestyle=line_style, ax=ax)
                
                if show_cdf:
                    sns.ecdfplot(df[column], ax=ax)
                
                if show_grid:
                    ax.grid(True)
                
                ax.set_xlabel(column)
                ax.set_ylabel("Density")
                ax.set_title(f"Density Plot of {column}")
                st.pyplot(fig)
            else:
                st.error("No numerical columns found in the uploaded file.")
        else:
            st.info("Please upload a CSV file to get started.")

# Tab 3: Pair Plot Generator
with tab3:
    # Streamlit app title
    st.title("Pair Plot Generator")

    # Sidebar for Pair Plot controls (only shown in this tab)
    st.sidebar.header("Pair Plot Settings")

    # Toggle between CSV upload and random data generation
    use_random_data = st.sidebar.checkbox("Use random data instead of uploading a file", key='pair_random_data')

    if use_random_data:
        # Random Data Settings
        num_features = st.sidebar.slider("Number of Features", min_value=2, max_value=10, value=3, key='pair_num_features')
        num_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=1000, value=200, key='pair_num_samples')
        
        # Select Distribution Type
        distribution_type = st.sidebar.selectbox("Select Distribution Type", ["Normal", "Uniform", "Exponential"], key='pair_distribution')
        
        if distribution_type == "Normal":
            mean_value = st.sidebar.number_input("Mean", value=0.0, key='pair_mean')
            std_dev = st.sidebar.number_input("Standard Deviation", value=1.0, key='pair_std_dev')
            data = np.random.normal(loc=mean_value, scale=std_dev, size=(num_samples, num_features))
        
        elif distribution_type == "Uniform":
            min_val = st.sidebar.number_input("Min Value", value=0.0, key='pair_min_val')
            max_val = st.sidebar.number_input("Max Value", value=1.0, key='pair_max_val')
            data = np.random.uniform(low=min_val, high=max_val, size=(num_samples, num_features))
        
        elif distribution_type == "Exponential":
            lambda_val = st.sidebar.number_input("Lambda (Rate Parameter)", value=1.0, key='pair_lambda')
            data = np.random.exponential(scale=1/lambda_val, size=(num_samples, num_features))

        # Create DataFrame
        column_names = [f"Feature {i+1}" for i in range(num_features)]
        df = pd.DataFrame(data, columns=column_names)

        # Select columns for Pair Plot
        selected_columns = st.multiselect("Select columns for Pair Plot", df.columns, default=df.columns[:2], key='pair_columns')

        if len(selected_columns) < 2:
            st.warning("Please select at least two numerical columns for the Pair Plot.")
        else:
            # Generate Pair Plot
            st.subheader("Pair Plot")
            g = sns.pairplot(df[selected_columns])
            st.pyplot(g.fig)
    else:
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key='pair_file_uploader')
        
        if uploaded_file is not None:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Detect numerical columns
            numerical_columns = df.select_dtypes(include=['number']).columns
            
            if len(numerical_columns) < 2:
                st.warning("The uploaded CSV file must contain at least two numerical columns for the Pair Plot.")
            else:
                # Select columns for Pair Plot
                selected_columns = st.multiselect("Select columns for Pair Plot", numerical_columns, default=numerical_columns[:2], key='csv_pair_columns')
                
                if len(selected_columns) < 2:
                    st.warning("Please select at least two numerical columns for the Pair Plot.")
                else:
                    # Generate Pair Plot
                    st.subheader("Pair Plot")
                    g = sns.pairplot(df[selected_columns])
                    st.pyplot(g.fig)
        else:
            st.info("Please upload a CSV file to get started.")

# Tab 4: Enhanced Joint Plot Generator
with tab4:
    # Streamlit app title
    st.title("Enhanced Joint Plot Generator")

    # Sidebar for Joint Plot controls (only shown in this tab)
    st.sidebar.header("Joint Plot Settings")

    # Toggle between CSV upload and random data generation
    use_random_data = st.sidebar.checkbox("Use random data instead of uploading a file", key='joint_random_data')

    # Color Palette Selection
    color_palette = st.sidebar.selectbox("Select Color Palette", sns.color_palette().as_hex(), key='joint_color_palette')

    # Grid Option
    show_grid = st.sidebar.checkbox("Show Grid Lines", value=True, key='joint_show_grid')

    # Regression Line Toggle
    show_regression = st.sidebar.checkbox("Show Regression Line (Only for Scatter)", value=False, key='joint_show_regression')

    if use_random_data:
        # Random Data Settings
        num_features = st.sidebar.slider("Number of Features", min_value=2, max_value=10, value=3, key='joint_num_features')
        num_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=1000, value=200, key='joint_num_samples')
        
        # Select Distribution Type
        distribution_type = st.sidebar.selectbox("Select Distribution Type", ["Normal", "Uniform", "Exponential"], key='joint_distribution')
        
        if distribution_type == "Normal":
            mean_value = st.sidebar.number_input("Mean", value=0.0, key='joint_mean')
            std_dev = st.sidebar.number_input("Standard Deviation", value=1.0, key='joint_std_dev')
            data = np.random.normal(loc=mean_value, scale=std_dev, size=(num_samples, num_features))
        
        elif distribution_type == "Uniform":
            min_val = st.sidebar.number_input("Min Value", value=0.0, key='joint_min_val')
            max_val = st.sidebar.number_input("Max Value", value=1.0, key='joint_max_val')
            data = np.random.uniform(low=min_val, high=max_val, size=(num_samples, num_features))
        
        elif distribution_type == "Exponential":
            lambda_val = st.sidebar.number_input("Lambda (Rate Parameter)", value=1.0, key='joint_lambda')
            data = np.random.exponential(scale=1/lambda_val, size=(num_samples, num_features))

        # Create DataFrame
        column_names = [f"Feature {i+1}" for i in range(num_features)]
        df = pd.DataFrame(data, columns=column_names)

        # Select columns for Joint Plot
        selected_columns = st.multiselect("Select two columns for Joint Plot", df.columns, default=df.columns[:2], key='joint_columns')

        if len(selected_columns) == 2:
            # Choose Joint Plot Type
            plot_type = st.sidebar.selectbox("Select Joint Plot Type", ["scatter", "kde", "hex"], key='joint_plot_type')
            
            # Generate Joint Plot
            st.subheader("Joint Plot")
            g = sns.jointplot(data=df, x=selected_columns[0], y=selected_columns[1], kind=plot_type, color=color_palette)
            
            if show_regression and plot_type == "scatter":
                g = sns.jointplot(data=df, x=selected_columns[0], y=selected_columns[1], kind="reg", color=color_palette)
            
            st.pyplot(g.fig)
        else:
            st.warning("Please select exactly two numerical columns for the Joint Plot.")
    else:
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key='joint_file_uploader')
        
        if uploaded_file is not None:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Detect numerical columns
            numerical_columns = df.select_dtypes(include=['number']).columns
            
            if len(numerical_columns) < 2:
                st.warning("The uploaded CSV file must contain at least two numerical columns for the Joint Plot.")
            else:
                # Select columns for Joint Plot
                selected_columns = st.multiselect("Select two columns for Joint Plot", numerical_columns, default=numerical_columns[:2], key='csv_joint_columns')
                
                if len(selected_columns) == 2:
                    # Choose Joint Plot Type
                    plot_type = st.sidebar.selectbox("Select Joint Plot Type", ["scatter", "kde", "hex"], key='csv_joint_plot_type')
                    
                    # Generate Joint Plot
                    st.subheader("Joint Plot")
                    g = sns.jointplot(data=df, x=selected_columns[0], y=selected_columns[1], kind=plot_type, color=color_palette)
                    
                    if show_regression and plot_type == "scatter":
                        g = sns.jointplot(data=df, x=selected_columns[0], y=selected_columns[1], kind="reg", color=color_palette)
                    
                    st.pyplot(g.fig)
                else:
                    st.warning("Please select exactly two numerical columns for the Joint Plot.")
        else:
            st.info("Please upload a CSV file to get started.")

# Required libraries
# pip install streamlit pandas numpy scipy matplotlib seaborn