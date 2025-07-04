"""
ApexAI Dashboard - Main Streamlit Application
Basic dashboard to get started with F1 data visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.data_pipeline.fastf1_client import FastF1Client
    from src.utils.visualization import F1Visualizer, prepare_lap_time_data
    from src.config.config import DashboardConfig
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure you're running from the project root directory")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title=DashboardConfig.PAGE_TITLE,
    page_icon=DashboardConfig.PAGE_ICON,
    layout=DashboardConfig.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for F1 styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1E1E;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #FF1E1E;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2E2E2E 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #FF1E1E;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #1E1E1E;
    }
    
    .stSelectbox > div > div {
        background-color: #2E2E2E;
        border: 1px solid #FF1E1E;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'f1_client' not in st.session_state:
    st.session_state.f1_client = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = F1Visualizer()
if 'current_data' not in st.session_state:
    st.session_state.current_data = {}

def initialize_client():
    """Initialize the FastF1 client"""
    try:
        st.session_state.f1_client = FastF1Client()
        return True
    except Exception as e:
        st.error(f"Failed to initialize F1 client: {e}")
        return False

def load_race_data(year: int, race_name: str, session_type: str):
    """Load race data for the selected parameters"""
    try:
        with st.spinner(f"Loading {session_type} data for {year} {race_name}..."):
            # Get the specific session
            import fastf1
            session = fastf1.get_session(year, race_name, session_type)
            session.load()
            
            # Extract different types of data
            results = st.session_state.f1_client.get_session_results(session)
            lap_data = st.session_state.f1_client.get_lap_data(session)
            tire_data = st.session_state.f1_client.get_tire_strategy_data(session)
            position_data = st.session_state.f1_client.get_position_data(session)
            
            # Store in session state
            st.session_state.current_data = {
                'session': session,
                'results': results,
                'lap_data': lap_data,
                'tire_data': tire_data,
                'position_data': position_data,
                'race_name': race_name,
                'year': year,
                'session_type': session_type
            }
            
            return True
            
    except Exception as e:
        st.error(f"Error loading race data: {e}")
        return False

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏁 ApexAI - F1 Race Intelligence</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for race selection
    with st.sidebar:
        st.header("📊 Race Selection")
        
        # Initialize client if not done
        if st.session_state.f1_client is None:
            if st.button("Initialize F1 Client"):
                if initialize_client():
                    st.success("F1 Client initialized successfully!")
                    st.rerun()
        
        if st.session_state.f1_client is not None:
            # Year selection
            year = st.selectbox(
                "Select Year",
                options=[2024, 2023, 2022],
                index=0
            )
            
            # Get available races for the year
            try:
                races = st.session_state.f1_client.get_available_races(year)
                if races:
                    race_name = st.selectbox(
                        "Select Race",
                        options=races,
                        index=0 if races else None
                    )
                    
                    # Session type selection
                    session_type = st.selectbox(
                        "Select Session",
                        options=["R", "Q", "FP3", "FP2", "FP1"],
                        index=0,
                        format_func=lambda x: {
                            "R": "🏁 Race",
                            "Q": "⚡ Qualifying", 
                            "FP3": "🏃 Free Practice 3",
                            "FP2": "🏃 Free Practice 2",
                            "FP1": "🏃 Free Practice 1"
                        }.get(x, x)
                    )
                    
                    # Load data button
                    if st.button("Load Race Data", type="primary"):
                        if load_race_data(year, race_name, session_type):
                            st.success(f"Loaded {session_type} data for {year} {race_name}")
                            st.rerun()
                
            except Exception as e:
                st.error(f"Error getting race list: {e}")
    
    # Main content area
    if st.session_state.current_data:
        data = st.session_state.current_data
        
        # Race info header
        st.subheader(f"📍 {data['year']} {data['race_name']} - {data['session_type']}")
        
        # Key metrics
        if not data['results'].empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Drivers", len(data['results']))
            
            with col2:
                if not data['lap_data'].empty:
                    total_laps = data['lap_data']['LapNumber'].max()
                    st.metric("Total Laps", total_laps)
            
            with col3:
                if not data['results'].empty and 'Q1' in data['results'].columns:
                    fastest_q1 = data['results']['Q1'].min()
                    st.metric("Fastest Lap", f"{fastest_q1:.3f}s" if pd.notna(fastest_q1) else "N/A")
            
            with col4:
                if not data['tire_data'].empty:
                    compounds_used = data['tire_data']['Compound'].nunique()
                    st.metric("Compounds Used", compounds_used)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Lap Times", "🏎️ Tire Strategy", "🏆 Positions", "📊 Analysis"])
        
        with tab1:
            st.header("Lap Times Analysis")
            
            if not data['lap_data'].empty:
                # Driver selection
                available_drivers = data['lap_data']['Driver'].unique()
                selected_drivers = st.multiselect(
                    "Select Drivers",
                    options=available_drivers,
                    default=available_drivers[:5] if len(available_drivers) >= 5 else available_drivers
                )
                
                if selected_drivers:
                    # Prepare and plot lap times
                    filtered_data = data['lap_data'][data['lap_data']['Driver'].isin(selected_drivers)]
                    clean_data = prepare_lap_time_data(filtered_data)
                    
                    fig = st.session_state.visualizer.plot_lap_times(
                        clean_data, 
                        selected_drivers,
                        f"Lap Times - {data['race_name']} {data['session_type']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show lap time statistics
                    st.subheader("Lap Time Statistics")
                    lap_stats = clean_data.groupby('Driver')['LapTime'].agg([
                        'min', 'mean', 'max', 'std'
                    ]).round(3)
                    lap_stats.columns = ['Fastest', 'Average', 'Slowest', 'Std Dev']
                    st.dataframe(lap_stats, use_container_width=True)
            else:
                st.info("No lap time data available for this session")
        
        with tab2:
            st.header("Tire Strategy Analysis")
            
            if not data['tire_data'].empty:
                # Tire strategy visualization
                fig = st.session_state.visualizer.plot_tire_strategy(
                    data['tire_data'],
                    f"Tire Strategy - {data['race_name']} {data['session_type']}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tire compound usage statistics
                st.subheader("Compound Usage")
                compound_stats = data['tire_data']['Compound'].value_counts()
                st.bar_chart(compound_stats)
                
                # Tire age analysis
                if 'TyreLife' in data['tire_data'].columns:
                    st.subheader("Tire Age Distribution")
                    tire_age_stats = data['tire_data'].groupby('Compound')['TyreLife'].describe()
                    st.dataframe(tire_age_stats, use_container_width=True)
            else:
                st.info("No tire strategy data available for this session")
        
        with tab3:
            st.header("Position Analysis")
            
            if not data['position_data'].empty:
                # Position changes plot
                fig = st.session_state.visualizer.plot_position_changes(
                    data['position_data'],
                    f"Position Changes - {data['race_name']} {data['session_type']}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Final results table
                if not data['results'].empty:
                    st.subheader("Session Results")
                    results_display = data['results'][['FullName', 'Position', 'Points']].copy()
                    results_display = results_display.sort_values('Position')
                    st.dataframe(results_display, use_container_width=True)
            else:
                st.info("No position data available for this session")
        
        with tab4:
            st.header("Advanced Analysis")
            
            # Data summary
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Data:**")
                st.write(f"- Results: {len(data['results'])} rows")
                st.write(f"- Lap Data: {len(data['lap_data'])} rows")
                st.write(f"- Tire Data: {len(data['tire_data'])} rows")
                st.write(f"- Position Data: {len(data['position_data'])} rows")
            
            with col2:
                if not data['lap_data'].empty:
                    st.write("**Lap Data Info:**")
                    st.write(f"- Total Laps: {data['lap_data']['LapNumber'].max()}")
                    st.write(f"- Drivers: {data['lap_data']['Driver'].nunique()}")
                    st.write(f"- Date: {data['lap_data']['date'].iloc[0] if 'date' in data['lap_data'].columns else 'N/A'}")
            
            # Raw data preview
            st.subheader("Raw Data Preview")
            data_type = st.selectbox(
                "Select Data Type",
                options=["results", "lap_data", "tire_data", "position_data"]
            )
            
            if data_type in data and not data[data_type].empty:
                st.dataframe(data[data_type].head(10), use_container_width=True)
            else:
                st.info(f"No {data_type} available")
    
    else:
        # Welcome screen
        st.header("🏎️ Welcome to ApexAI")
        st.write("""
        ApexAI is your F1 race intelligence system. Get started by:
        
        1. **Initialize the F1 Client** in the sidebar
        2. **Select a race** from the dropdown menus
        3. **Load the race data** to start analyzing
        
        ### Features:
        - 📈 **Lap Time Analysis**: Compare driver performance
        - 🏎️ **Tire Strategy**: Visualize compound usage and degradation
        - 🏆 **Position Tracking**: See how positions change throughout the race
        - 📊 **Advanced Analysis**: Dive deep into the data
        
        ### Data Sources:
        - FastF1 API for telemetry and timing data
        - Ergast API for historical race information
        """)
        
        # Show sample visualization
        st.subheader("Sample Visualization")
        
        # Create sample data for demo
        sample_data = pd.DataFrame({
            'LapNumber': range(1, 21),
            'Driver': ['HAM'] * 20,
            'LapTime': np.random.normal(90, 2, 20)
        })
        
        fig = st.session_state.visualizer.plot_lap_times(
            sample_data, 
            ['HAM'],
            "Sample Lap Times (Demo Data)"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()