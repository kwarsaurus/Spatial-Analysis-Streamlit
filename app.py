import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import json
from paste import restaurant_recomendation_v1

# Page configuration
st.set_page_config(
    page_title="Restaurant Location ML System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_system():
    """Load ML system with caching"""
    try:
        return RestaurantLocationML('final_model_updated')
    except Exception as e:
        st.error(f"Error loading ML system: {e}")
        st.error("Please ensure 'final_model_updated' directory exists with all required files.")
        return None

@st.cache_data
def load_portfolio_data(ml_system):
    """Load portfolio data with caching"""
    if ml_system:
        return ml_system.analyze_portfolio()
    return None

def create_location_map(locations, scores=None):
    """Create interactive map with location markers"""
    # Center map on Jakarta Selatan
    center_lat, center_lng = -6.235, 106.805
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for scores
    def get_color(score):
        if score >= 0.3: return 'green'
        elif score >= 0.2: return 'orange'
        elif score >= 0.1: return 'red'
        else: return 'darkred'
    
    # Add markers
    for i, (lat, lng, district, category) in enumerate(locations):
        score = scores[i] if scores else 0
        color = get_color(score)
        
        folium.Marker(
            [lat, lng],
            popup=f"""
            <b>{district}</b><br>
            Category: {category}<br>
            Score: {score:.3f}
            """,
            tooltip=f"{district} - {category}",
            icon=folium.Icon(color=color, icon='utensils', prefix='fa')
        ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Restaurant Location ML System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load ML System
    ml_system = load_ml_system()
    
    if not ml_system:
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Dashboard", "📍 Location Scoring", "📊 Portfolio Analysis", "🔍 Location Comparison", "📋 Expansion Report"]
    )
    
    # Dashboard
    if page == "🏠 Dashboard":
        st.header("📊 System Overview")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Spatial Model R²",
                value="0.451",
                delta="Medium Accuracy"
            )
        
        with col2:
            st.metric(
                label="Portfolio Model R²", 
                value="0.464",
                delta="Good Performance"
            )
        
        with col3:
            st.metric(
                label="Active Landmarks",
                value="5+",
                delta="Jakarta Selatan"
            )
        
        with col4:
            st.metric(
                label="Model Status",
                value="Ready",
                delta="✅ Loaded"
            )
        
        st.markdown("---")
        
        # Quick portfolio overview
        portfolio_data = load_portfolio_data(ml_system)
        if portfolio_data:
            st.subheader("📈 Portfolio Quick Stats")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Branches",
                    portfolio_data['portfolio_summary']['total_branches']
                )
            
            with col2:
                st.metric(
                    "Avg Performance",
                    f"{portfolio_data['portfolio_summary']['avg_performance']:.3f}"
                )
            
            with col3:
                st.metric(
                    "Optimization Opportunities",
                    len(portfolio_data['optimization_candidates'])
                )
        
        # System capabilities
        st.subheader("🎯 System Capabilities")
        
        capabilities = [
            "🗺️ **Spatial Analysis**: Distance to key landmarks and business districts",
            "🏢 **Competition Analysis**: Competitor density and market intensity",
            "📊 **Portfolio Optimization**: Identify underperforming branches",
            "🔍 **Location Comparison**: Compare multiple candidate locations",
            "📋 **Expansion Planning**: Generate comprehensive expansion reports"
        ]
        
        for capability in capabilities:
            st.markdown(capability)
    
    # Location Scoring
    elif page == "📍 Location Scoring":
        st.header("📍 Score New Location")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Location Details")
            
            # Input fields
            latitude = st.number_input(
                "Latitude",
                value=-6.235,
                min_value=-6.5,
                max_value=-6.0,
                step=0.001,
                format="%.6f"
            )
            
            longitude = st.number_input(
                "Longitude", 
                value=106.805,
                min_value=106.5,
                max_value=107.0,
                step=0.001,
                format="%.6f"
            )
            
            # District options (common Jakarta Selatan districts)
            districts = [
                'Kebayoran Baru', 'Setia Budi', 'Cilandak', 'Kebayoran Lama',
                'Mampang Prapatan', 'Pancoran', 'Jagakarsa', 'Pasar Minggu'
            ]
            district = st.selectbox("District", districts)
            
            # Category options
            categories = [
                'Fresh Juices', 'Coffee and Hot Beverages', 'Salads and Bowls',
                'Japanese Cuisine', 'Italian Cuisine', 'Local Indonesian'
            ]
            category = st.selectbox("Restaurant Category", categories)
            
            # Score button
            if st.button("🎯 Score Location", type="primary"):
                with st.spinner("Analyzing location..."):
                    result = ml_system.score_new_location(latitude, longitude, district, category)
                    st.session_state['location_result'] = result
        
        with col2:
            # Map visualization
            st.subheader("📍 Location Map")
            location_map = create_location_map([(latitude, longitude, district, category)])
            folium_static(location_map, width=600, height=400)
        
        # Display results
        if 'location_result' in st.session_state:
            result = st.session_state['location_result']
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Score display
            score = result['spatial_score']['score']
            score_level = result['spatial_score']['level']
            
            # Color coding for recommendation
            if score >= 0.3:
                box_class = "success-box"
            elif score >= 0.2:
                box_class = "warning-box"
            else:
                box_class = "danger-box"
            
            st.markdown(f"""
            <div class="recommendation-box {box_class}">
                <h3>Score: {score:.3f} ({score_level})</h3>
                <p><strong>{result['recommendation']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key factors
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Distance to Kemang",
                    f"{result['key_factors']['distance_to_kemang_km']} km"
                )
                
            with col2:
                st.metric(
                    "Distance to CBD",
                    f"{result['key_factors']['distance_to_cbd_km']} km"
                )
                
            with col3:
                st.metric(
                    "Competitors (1km)",
                    result['key_factors']['competitors_1km']
                )
            
            # Spatial insights
            if result['spatial_insights']:
                st.subheader("💡 Spatial Insights")
                for insight in result['spatial_insights']:
                    st.write(f"• {insight}")
    
    # Portfolio Analysis
    elif page == "📊 Portfolio Analysis":
        st.header("📊 Portfolio Analysis")
        
        portfolio_data = load_portfolio_data(ml_system)
        
        if portfolio_data:
            # Summary metrics
            summary = portfolio_data['portfolio_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Branches", summary['total_branches'])
            with col2:
                st.metric("Avg Performance", f"{summary['avg_performance']:.3f}")
            with col3:
                st.metric("Avg Potential", f"{summary['avg_potential']:.3f}")
            with col4:
                st.metric("Optimization Gap", f"{summary['optimization_gap']:.3f}")
            
            st.markdown("---")
            
            # Performance distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Branch Status Distribution")
                status_dist = portfolio_data['status_distribution']
                
                # Create pie chart
                fig = px.pie(
                    values=list(status_dist.values()),
                    names=list(status_dist.keys()),
                    title="Branch Performance Status"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🏆 Top Performers")
                top_performers = portfolio_data['top_performers']
                
                df_top = pd.DataFrame(top_performers)
                if not df_top.empty:
                    st.dataframe(df_top, use_container_width=True)
            
            # District analysis
            if 'district_insights' in portfolio_data:
                st.subheader("🗺️ Performance by District")
                district_data = portfolio_data['district_insights']
                
                df_district = pd.DataFrame(district_data).T.reset_index()
                df_district.columns = ['District', 'Avg Performance', 'Performance Gap']
                
                fig = px.bar(
                    df_district,
                    x='District',
                    y='Avg Performance',
                    title="Average Performance by District",
                    color='Performance Gap',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Optimization candidates
            st.subheader("⚠️ Optimization Candidates")
            opt_candidates = portfolio_data['optimization_candidates']
            
            if opt_candidates:
                df_opt = pd.DataFrame(opt_candidates)
                st.dataframe(df_opt, use_container_width=True)
            else:
                st.info("No branches requiring immediate optimization.")
    
    # Location Comparison
    elif page == "🔍 Location Comparison":
        st.header("🔍 Compare Multiple Locations")
        
        st.write("Compare up to 5 potential locations side by side.")
        
        # Input for multiple locations
        locations = []
        
        num_locations = st.slider("Number of locations to compare", 2, 5, 3)
        
        for i in range(num_locations):
            st.subheader(f"📍 Location {i+1}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                lat = st.number_input(f"Latitude {i+1}", value=-6.235-i*0.01, key=f"lat_{i}")
            with col2:
                lng = st.number_input(f"Longitude {i+1}", value=106.805+i*0.01, key=f"lng_{i}")
            with col3:
                districts = ['Kebayoran Baru', 'Setia Budi', 'Cilandak', 'Kebayoran Lama']
                district = st.selectbox(f"District {i+1}", districts, key=f"district_{i}")
            with col4:
                categories = ['Fresh Juices', 'Coffee and Hot Beverages', 'Salads and Bowls']
                category = st.selectbox(f"Category {i+1}", categories, key=f"category_{i}")
            
            locations.append((lat, lng, district, category))
        
        if st.button("🔍 Compare Locations", type="primary"):
            with st.spinner("Comparing locations..."):
                comparison_results = ml_system.compare_locations(locations)
                st.session_state['comparison_results'] = comparison_results
        
        # Display comparison results
        if 'comparison_results' in st.session_state:
            results = st.session_state['comparison_results']
            
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
            # Create comparison table
            comparison_data = []
            for i, result in enumerate(results):
                comparison_data.append({
                    'Rank': i + 1,
                    'District': result['location']['district'],
                    'Category': result['location']['category'],
                    'Score': result['spatial_score']['score'],
                    'Level': result['spatial_score']['level'],
                    'Distance to Kemang (km)': result['key_factors']['distance_to_kemang_km'],
                    'Competitors (1km)': result['key_factors']['competitors_1km']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                df_comparison,
                x='District',
                y='Score',
                color='Level',
                title="Location Scores Comparison",
                text='Score'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Map with all locations
            st.subheader("🗺️ Locations Map")
            scores = [r['spatial_score']['score'] for r in results]
            comparison_map = create_location_map(locations, scores)
            folium_static(comparison_map, width=800, height=500)
    
    # Expansion Report
    elif page == "📋 Expansion Report":
        st.header("📋 Expansion Planning Report")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Report Parameters")
            
            target_branches = st.slider("Target New Branches", 1, 10, 3)
            
            all_categories = ['Fresh Juices', 'Coffee and Hot Beverages', 'Salads and Bowls', 
                             'Japanese Cuisine', 'Italian Cuisine', 'Local Indonesian']
            
            focus_categories = st.multiselect(
                "Focus Categories",
                all_categories,
                default=['Fresh Juices', 'Coffee and Hot Beverages']
            )
            
            if st.button("📋 Generate Report", type="primary"):
                with st.spinner("Generating comprehensive expansion report..."):
                    try:
                        report = ml_system.generate_expansion_report(
                            target_branches=target_branches,
                            focus_categories=focus_categories
                        )
                        st.session_state['expansion_report'] = report
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
        
        with col2:
            if 'expansion_report' in st.session_state:
                report = st.session_state['expansion_report']
                
                if 'executive_summary' in report:
                    st.subheader("📋 Executive Summary")
                    
                    exec_summary = report['executive_summary']
                    
                    st.write(f"**Target branches:** {exec_summary.get('target_new_branches', 'N/A')}")
                    st.write(f"**Recommended districts:** {', '.join(exec_summary.get('recommended_districts', []))}")
                    st.write(f"**Investment estimate:** {exec_summary.get('total_investment_estimate', 'N/A')}")
                    
                    # Risk assessment
                    if 'risk_assessment' in report:
                        st.subheader("⚠️ Risk Assessment")
                        risk = report['risk_assessment']
                        
                        col1_risk, col2_risk = st.columns(2)
                        with col1_risk:
                            st.write(f"**Model confidence:** {risk.get('model_confidence', 'N/A')}")
                            st.write(f"**Market risk:** {risk.get('market_risk', 'N/A')}")
                        with col2_risk:
                            st.write(f"**Competition risk:** {risk.get('competition_risk', 'N/A')}")
                            st.write(f"**Location risk:** {risk.get('location_risk', 'N/A')}")
                    
                    # Next steps
                    if 'next_steps' in report:
                        st.subheader("📝 Next Steps")
                        for step in report['next_steps']:
                            st.write(f"• {step}")
                    
                    # Location recommendations
                    if 'location_recommendations' in report:
                        st.subheader("📍 Location Recommendations")
                        
                        for i, rec in enumerate(report['location_recommendations'], 1):
                            with st.expander(f"Location {i}: {rec['location']['district']}"):
                                st.write(f"**Coordinates:** {rec['location']['coordinates']}")
                                st.write(f"**Category:** {rec['location']['category']}")
                                st.write(f"**Score:** {rec['spatial_score']['score']:.3f}")
                                st.write(f"**Recommendation:** {rec['recommendation']}")
                else:
                    st.error("Report generation failed. Please check your model files.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🍽️ **Restaurant Location ML System** | Built with Streamlit | "
        "Powered by Spatial Analysis & Machine Learning"
    )

if __name__ == "__main__":
    main()
