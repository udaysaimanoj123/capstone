import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    # Replace this with your actual data loading mechanism
    df = pd.read_csv('dataset.csv')
    return df

def train_model(df):
    # Data preprocessing
    df_encoded = pd.get_dummies(df, columns=['Dist Name'], drop_first=True)
    X = df_encoded.drop(['RICE PRODUCTION (1000 tons)', 'RICE YIELD (Kg per ha)'], axis=1)
    y = df_encoded['RICE PRODUCTION (1000 tons)']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bootstrap Sampling
    train_data = pd.concat([X_train, y_train], axis=1)
    augmented_data = resample(train_data, replace=True, n_samples=len(train_data) * 2, random_state=42)
    X_train_resampled = augmented_data.drop(['RICE PRODUCTION (1000 tons)'], axis=1)
    y_train_resampled = augmented_data['RICE PRODUCTION (1000 tons)']
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, X.columns, X_train, X_test, y_train, y_test

def create_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                    color_continuous_scale='RdBu',
                    title='Correlation Heatmap of Numerical Features')
    return fig

def create_production_trend(df):
    yearly_prod = df.groupby('Year')['RICE PRODUCTION (1000 tons)'].mean().reset_index()
    fig = px.line(yearly_prod, x='Year', y='RICE PRODUCTION (1000 tons)',
                  title='Average Rice Production Trend Over Years')
    return fig

def create_district_comparison(df):
    district_stats = df.groupby('Dist Name')['RICE PRODUCTION (1000 tons)'].agg(['mean', 'std']).reset_index()
    fig = px.bar(district_stats, x='Dist Name', y='mean',
                 error_y='std',
                 title='Average Rice Production by District (with Standard Deviation)')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_scatter_matrix(df):
    features = ['precipitation', 'temperature_maximum', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)']
    fig = px.scatter_matrix(df[features],
                           title='Feature Relationships Matrix')
    return fig

def display_model_metrics(y_test, y_pred):
    metrics = {
        'Mean Squared Error': mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
        'R² Score': r2_score(y_test, y_pred)
    }
    return metrics

def main():
    st.set_page_config(layout="wide")
    st.title('Rice Production Analysis and Prediction')
    st.write("""
    This application provides comprehensive analysis and prediction of rice production based on various environmental 
    and geographical factors.
    """)
    
    try:
        # Load and train model
        df = load_data()
        model, feature_names, X_train, X_test, y_train, y_test = train_model(df)
        
        # Create tabs for different sections
        tabs = st.tabs(['Data Analysis', 'Prediction', 'Model Performance'])
        
        # Data Analysis Tab
        with tabs[0]:
            st.header('Data Analysis')
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Correlation Heatmap')
                fig_corr = create_correlation_heatmap(df)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.subheader('Production Trends')
                fig_trend = create_production_trend(df)
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                st.subheader('District-wise Production')
                fig_dist = create_district_comparison(df)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.subheader('Feature Relationships')
                fig_scatter = create_scatter_matrix(df)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Display basic statistics
            st.subheader('Statistical Summary')
            st.write(df.describe())
        
        # Prediction Tab
        with tabs[1]:
            st.header('Prediction Interface')
            col1, col2 = st.columns(2)
            
            with col1:
                year = st.number_input('Year', min_value=2000, max_value=2030, value=2024)
                precipitation = st.number_input('Precipitation', min_value=0.0, max_value=500.0, value=92.12)
                evapotranspiration_potential = st.number_input('Evapotranspiration Potential', min_value=0.0, max_value=500.0, value=128.20)
                evapotranspiration_actual = st.number_input('Evapotranspiration Actual', min_value=0.0, max_value=500.0, value=82.79)
                temperature_maximum = st.number_input('Maximum Temperature', min_value=0.0, max_value=50.0, value=30.80)
            
            with col2:
                temperature_minimum = st.number_input('Minimum Temperature', min_value=0.0, max_value=40.0, value=23.10)
                water_deficiet = st.number_input('Water Deficit', min_value=0.0, max_value=200.0, value=46.89)
                rice_area = st.number_input('Rice Area (1000 ha)', min_value=0.0, max_value=1000.0, value=192.99)
            
            districts = ['Chittoor', 'East Godavari', 'Guntur', 'Kadapa YSR', 'Krishna', 
                        'Kurnool', 'Prakasam', 'S.P.S.Nellore', 'Srikakulam', 
                        'Visakhapatnam', 'Vizianagaram', 'West Godavari']
            selected_district = st.selectbox('Select District', districts)
            
            if st.button('Predict Rice Production'):
                input_data = {
                    'Year': [year],
                    'precipitation': [precipitation],
                    'evapotranspiration_potential': [evapotranspiration_potential],
                    'evapotranspiration_actual': [evapotranspiration_actual],
                    'temperature_maximum': [temperature_maximum],
                    'temperature_minimum': [temperature_minimum],
                    'water_deficiet': [water_deficiet],
                    'RICE AREA (1000 ha)': [rice_area]
                }
                
                for district in districts:
                    input_data[f'Dist Name_{district}'] = [1 if district == selected_district else 0]
                
                input_df = pd.DataFrame(input_data)
                input_df = input_df.reindex(columns=feature_names, fill_value=0)
                
                prediction = model.predict(input_df)
                
                st.success(f'Predicted Rice Production: {prediction[0]:,.2f} thousand tons')
                
                # Historical comparison
                district_avg = df[df['Dist Name'] == selected_district]['RICE PRODUCTION (1000 tons)'].mean()
                st.info(f"""
                Historical Context:
                - Average production in {selected_district}: {district_avg:,.2f} thousand tons
                - Prediction is {(prediction[0] - district_avg) / district_avg * 100:.1f}% different from historical average
                """)
        
        # Model Performance Tab
        with tabs[2]:
            st.header('Model Performance Analysis')
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            metrics = display_model_metrics(y_test, y_pred)
            
            # Display metrics in columns
            cols = st.columns(len(metrics))
            for col, (metric, value) in zip(cols, metrics.items()):
                col.metric(metric, f"{value:.4f}")
            
            # Actual vs Predicted Plot
            fig = px.scatter(x=y_test, y=y_pred,
                           labels={'x': 'Actual Production', 'y': 'Predicted Production'},
                           title='Actual vs Predicted Production')
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines',
                                   name='Perfect Prediction',
                                   line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': abs(model.coef_)
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Feature', y='Importance',
                        title='Feature Importance in Prediction Model')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the dataset file is in the correct location and format.")

if __name__ == '__main__':
    main()