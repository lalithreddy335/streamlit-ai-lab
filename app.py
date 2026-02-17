"""
Streamlit Exploratory Data Analysis (EDA) Application
This app demonstrates data exploration using the Iris dataset
Features: data preview, summary statistics, histograms, and scatter plots
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# Configure the page
st.set_page_config(
    page_title="Iris Dataset EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title and description
st.title("📊 Exploratory Data Analysis - Iris Dataset")
st.markdown("""
This application allows you to explore the classic Iris dataset with interactive visualizations.
You can analyze different numeric columns and create histograms and scatter plots.
""")

# Load the Iris dataset
@st.cache_data
def load_iris_data():
    """Load and prepare the Iris dataset"""
    iris = load_iris()
    # Create a DataFrame with the iris features and target
    iris_df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    # Add the target variable (species)
    iris_df['species'] = iris.target_names[iris.target]
    return iris_df

# Load data
iris_dataframe = load_iris_data()

# Display section: Dataset Overview
st.header("1. Dataset Overview")

# Create two columns for dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Rows", value=len(iris_dataframe))
with col2:
    st.metric(label="Total Columns", value=len(iris_dataframe.columns))
with col3:
    st.metric(label="Species", value=iris_dataframe['species'].nunique())

# Display the first few rows
st.subheader("First 10 rows of the dataset")
st.dataframe(iris_dataframe.head(10), use_container_width=True)

# Display section: Summary Statistics
st.header("2. Summary Statistics")
st.subheader("Statistical overview of numeric columns")
st.dataframe(iris_dataframe.describe(), use_container_width=True)

# Display section: Column Distribution
st.header("3. Column Visualization")

# Get numeric columns (exclude species which is categorical)
numeric_columns = iris_dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Sidebar controls for visualization
st.sidebar.header("Visualization Options")

# Allow user to select a numeric column for histogram
selected_column_histogram = st.sidebar.selectbox(
    label="Select column for histogram",
    options=numeric_columns,
    help="Choose a numeric column to visualize its distribution"
)

# Allow user to select two columns for scatter plot
scatter_x_column = st.sidebar.selectbox(
    label="Select X-axis for scatter plot",
    options=numeric_columns,
    index=0,
    help="Choose the first column for scatter plot"
)

scatter_y_column = st.sidebar.selectbox(
    label="Select Y-axis for scatter plot",
    options=numeric_columns,
    index=1,
    help="Choose the second column for scatter plot"
)

# Create two columns for visualizations
viz_col1, viz_col2 = st.columns(2)

# Create histogram
with viz_col1:
    st.subheader(f"Histogram: {selected_column_histogram}")
    histogram_fig = px.histogram(
        iris_dataframe,
        x=selected_column_histogram,
        nbins=20,
        color='species',
        title=f"Distribution of {selected_column_histogram}",
        labels={selected_column_histogram: selected_column_histogram}
    )
    histogram_fig.update_layout(height=400)
    st.plotly_chart(histogram_fig, use_container_width=True)

# Create scatter plot
with viz_col2:
    st.subheader(f"Scatter Plot: {scatter_x_column} vs {scatter_y_column}")
    scatter_fig = px.scatter(
        iris_dataframe,
        x=scatter_x_column,
        y=scatter_y_column,
        color='species',
        title=f"{scatter_x_column} vs {scatter_y_column}",
        labels={
            scatter_x_column: scatter_x_column,
            scatter_y_column: scatter_y_column
        }
    )
    scatter_fig.update_layout(height=400)
    st.plotly_chart(scatter_fig, use_container_width=True)

# Display section: Species Analysis
st.header("4. Species Distribution")

# Count of each species
species_counts = iris_dataframe['species'].value_counts()
species_fig = px.bar(
    x=species_counts.index,
    y=species_counts.values,
    labels={'x': 'Species', 'y': 'Count'},
    title='Distribution of Species',
    color=species_counts.index
)
st.plotly_chart(species_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**About this app:** Built with Streamlit | Data: Iris Dataset (150 samples, 4 features)
""")

