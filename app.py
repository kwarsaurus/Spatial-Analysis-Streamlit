import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import json

# Import your ML system
try:
    from restaurant_ml_system import RestaurantLocationML
except ImportError:
    try:
        from paste import RestaurantLocationML
    except ImportError:
        st.error("❌ Cannot import RestaurantLocationML class. Please ensure:")
        st.error("1. Your ML system file is named 'restaurant_ml_system.py' or 'paste.py'")
        st.error("2. The file contains the RestaurantLocationML class")
        st.error("3. The file is in the same directory as this Streamlit app")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Restaurant Location ML System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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

def load_ml_system():
    """Load ML system"""
    try:
        return RestaurantLocationML('final_model_updated')
    except FileNotFoundError:
        st.error("❌ **Model files not found!**")
        st.error("Please ensure the 'final_model_updated' directory exists with all required files")
        return None
    except Exception as e:
        st.error(f"❌ Error loading ML system: {e}")
        return None

def load_portfolio_data(ml_system):
    """Load portfolio data"""
    if ml_system:
        try:
            return ml_system.analyze_portfolio()
        except Exception as e:
            st.warning(f"Using demo data - Original error: {e}")
            # Return dummy data for demo
            return create_dummy_portfolio_data()
    return None

def create_dummy_portfolio_data():
    """Create dummy portfolio data for demo"""
    return {
        'portfolio_summary': {
            'total_branches': 15,
            'avg_performance': 0.342,
            'avg_potential': 0.356,
            'optimization_gap': 0.14
        },
        'status_distribution': {
            'Overperforming': 5,
            'As Expected': 7,
            'Underperforming': 3
        },
        'top_performers': [
            {'branchName': 'Kemang Raya', 'districtName': 'Kemang', 'performance_score': 0.485},
            {'branchName': 'SCBD Plaza', 'districtName': 'Setia Budi', 'performance_score': 0.467},
            {'branchName': 'Senayan City', 'districtName': 'Kebayoran Baru', 'performance_score': 0.441},
            {'branchName': 'Blok M Square', 'districtName': 'Kebayoran Baru', 'performance_score': 0.423},
            {'branchName': 'Pondok Indah Mall', 'districtName': 'Cilandak', 'performance_score': 0.402}
        ],
        'optimization_candidates': [
            {'branchName': 'Fatmawati', 'districtName': 'Cilandak', 'performance_gap': -0.087},
            {'branchName': 'Tebet Raya', 'districtName': 'Tebet', 'performance_gap': -0.056},
            {'branchName': 'Warung Buncit', 'districtName': 'Mampang', 'performance_gap': -0.034}
        ],
        'district_insights': {
            'Kemang': {'performance_score': 0.445, 'performance_gap': 0.023},
            'Setia Budi': {'performance_score': 0.412, 'performance_gap': 0.018},
            'Kebayoran Baru': {'performance_score': 0.389, 'performance_gap': 0.012},
            'Cilandak': {'performance_score': 0.298, 'performance_gap': -0.045},
            'Mampang': {'performance_score': 0.267, 'performance_gap': -0.067}
        },
        'category_insights': {
            'Fresh Juices': {'performance_score': 0.378, 'performance_gap': 0.015},
            'Coffee and Hot Beverages': {'performance_score': 0.356, 'performance_gap': 0.008},
            'Salads and Bowls': {'performance_score': 0.334, 'performance_gap': -0.012},
            'Japanese Cuisine': {'performance_score': 0.289, 'performance_gap': -0.034}
        }
    }

def create_location_map(locations, scores=None):
    """Create interactive map with location markers"""
    center_lat, center_lng = -6.235, 106.805
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='OpenStreetMap')
    
    def get_color(score):
        if score >= 0.3: return 'green'
        elif score >= 0.2: return 'orange'
        elif score >= 0.1: return 'red'
        else: return 'darkred'
    
    for i, (lat, lng, district, category) in enumerate(locations):
        score = scores[i] if scores else 0
        color = get_color(score)
        
        folium.Marker(
            [lat, lng],
            popup=f"<b>{district}</b><br>Category: {category}<br>Score: {score:.3f}",
            tooltip=f"{district} - {category}",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ Restaurant Location ML System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize ML System in session state
    if 'ml_system' not in st.session_state:
        with st.spinner("🔄 Loading ML System..."):
            st.session_state.ml_system = load_ml_system()
    
    ml_system = st.session_state.ml_system
    
    if not ml_system:
        st.error("❌ **Cannot proceed without ML system loaded.**")
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Dashboard", "📍 Location Scoring", "📊 Portfolio Analysis", "🔍 Location Comparison", "📋 Expansion Report"]
    )
    
    # Add refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        for key in ['ml_system', 'portfolio_data', 'portfolio_analysis']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Dashboard
    if page == "🏠 Dashboard":
        st.header("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spatial Model R²", "0.451", "Medium Accuracy")
        with col2:
            st.metric("Portfolio Model R²", "0.464", "Good Performance")
        with col3:
            st.metric("Active Landmarks", "5+", "Jakarta Selatan")
        with col4:
            st.metric("Model Status", "Ready", "✅ Loaded")
        
        st.markdown("---")
        
        # Quick portfolio overview
        if 'portfolio_data' not in st.session_state:
            with st.spinner("Loading portfolio data..."):
                st.session_state.portfolio_data = load_portfolio_data(ml_system)
        
        portfolio_data = st.session_state.portfolio_data
        
        if portfolio_data:
            st.subheader("📈 Portfolio Quick Stats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Branches", portfolio_data['portfolio_summary']['total_branches'])
            with col2:
                st.metric("Avg Performance", f"{portfolio_data['portfolio_summary']['avg_performance']:.3f}")
            with col3:
                st.metric("Optimization Opportunities", len(portfolio_data['optimization_candidates']))
            
            # Interactive Dashboard Charts
            st.subheader("📊 Interactive Analytics")
            
            # Performance distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Branch Status Distribution**")
                status_data = portfolio_data['status_distribution']
                fig_pie = px.pie(
                    values=list(status_data.values()),
                    names=list(status_data.keys()),
                    color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.write("**District Performance Comparison**")
                if 'district_insights' in portfolio_data:
                    district_data = portfolio_data['district_insights']
                    districts = list(district_data.keys())
                    scores = [data['performance_score'] for data in district_data.values()]
                    
                    fig_bar = px.bar(
                        x=districts, y=scores,
                        labels={'x': 'District', 'y': 'Performance Score'},
                        color=scores,
                        color_continuous_scale='Viridis'
                    )
                    fig_bar.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("⚠️ Portfolio data not available - using demo mode")
        
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
            
            latitude = st.number_input("Latitude", value=-6.235, min_value=-6.5, max_value=-6.0, step=0.001, format="%.6f")
            longitude = st.number_input("Longitude", value=106.805, min_value=106.5, max_value=107.0, step=0.001, format="%.6f")
            
            districts = ['Kebayoran Baru', 'Setia Budi', 'Cilandak', 'Kebayoran Lama', 'Mampang Prapatan', 'Pancoran']
            district = st.selectbox("District", districts)
            
            categories = ['Fresh Juices', 'Coffee and Hot Beverages', 'Salads and Bowls', 'Japanese Cuisine', 'Italian Cuisine']
            category = st.selectbox("Restaurant Category", categories)
            
            if st.button("🎯 Score Location", type="primary"):
                try:
                    with st.spinner("Analyzing location..."):
                        result = ml_system.score_new_location(latitude, longitude, district, category)
                        st.session_state['location_result'] = result
                        st.success("✅ Location scored successfully!")
                except Exception as e:
                    st.error(f"❌ Error scoring location: {str(e)}")
        
        with col2:
            st.subheader("📍 Location Map")
            location_map = create_location_map([(latitude, longitude, district, category)])
            folium_static(location_map, width=600, height=400)
        
        # Display results
        if 'location_result' in st.session_state:
            result = st.session_state['location_result']
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            score = result['spatial_score']['score']
            score_level = result['spatial_score']['level']
            
            if score >= 0.3: box_class = "success-box"
            elif score >= 0.2: box_class = "warning-box"
            else: box_class = "danger-box"
            
            st.markdown(f"""
            <div class="recommendation-box {box_class}">
                <h3>Score: {score:.3f} ({score_level})</h3>
                <p><strong>{result['recommendation']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Distance to Kemang", f"{result['key_factors']['distance_to_kemang_km']} km")
            with col2:
                st.metric("Distance to CBD", f"{result['key_factors']['distance_to_cbd_km']} km")
            with col3:
                st.metric("Competitors (1km)", result['key_factors']['competitors_1km'])
            
            if result['spatial_insights']:
                st.subheader("💡 Spatial Insights")
                for insight in result['spatial_insights']:
                    st.write(f"• {insight}")
    
    # Portfolio Analysis
    elif page == "📊 Portfolio Analysis":
        st.header("📊 Portfolio Analysis")
        
        if 'portfolio_analysis' not in st.session_state:
            with st.spinner("🔄 Analyzing portfolio..."):
                st.session_state.portfolio_analysis = load_portfolio_data(ml_system)
        
        portfolio_data = st.session_state.portfolio_analysis
        
        if not portfolio_data:
            st.error("❌ Unable to load portfolio data")
            return
        
        # Summary metrics with better styling
        st.subheader("📋 Portfolio Summary")
        summary = portfolio_data['portfolio_summary']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Branches", summary['total_branches'], help="Total number of active branches")
        with col2:
            st.metric("Avg Performance", f"{summary['avg_performance']:.3f}", help="Average performance score across all branches")
        with col3:
            st.metric("Avg Potential", f"{summary['avg_potential']:.3f}", help="Predicted potential performance")
        with col4:
            delta_gap = summary['optimization_gap']
            st.metric("Optimization Gap", f"{delta_gap:.3f}", delta=f"{delta_gap:.3f}", help="Total optimization opportunity")
        
        st.markdown("---")
        
        # Main analytics section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Branch Performance Status")
            status_dist = portfolio_data['status_distribution']
            
            # Create pie chart with custom colors
            fig = px.pie(
                values=list(status_dist.values()),
                names=list(status_dist.keys()),
                title="Distribution of Branch Performance",
                color_discrete_map={
                    'Overperforming': '#28a745',
                    'As Expected': '#17a2b8',
                    'Underperforming': '#dc3545'
                }
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏆 Top Performing Branches")
            top_performers = portfolio_data['top_performers']
            
            if top_performers:
                # Create a nice dataframe display
                df_top = pd.DataFrame(top_performers)
                df_top['Rank'] = range(1, len(df_top) + 1)
                df_top = df_top[['Rank', 'branchName', 'districtName', 'performance_score']]
                df_top.columns = ['Rank', 'Branch Name', 'District', 'Score']
                
                # Style the dataframe
                st.dataframe(
                    df_top,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Performance Score",
                            help="Branch performance score",
                            min_value=0,
                            max_value=1,
                            format="%.3f"
                        )
                    }
                )
            else:
                st.info("No top performers data available")
        
        # District Performance Analysis
        if 'district_insights' in portfolio_data:
            st.subheader("🗺️ Performance Analysis by District")
            
            district_data = portfolio_data['district_insights']
            df_district = pd.DataFrame(district_data).T.reset_index()
            df_district.columns = ['District', 'Avg Performance', 'Performance Gap']
            
            # Sort by performance
            df_district = df_district.sort_values('Avg Performance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance bar chart
                fig = px.bar(
                    df_district,
                    x='District',
                    y='Avg Performance',
                    title="Average Performance by District",
                    color='Avg Performance',
                    color_continuous_scale='Viridis',
                    text='Avg Performance'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance gap analysis
                fig = px.bar(
                    df_district,
                    x='District',
                    y='Performance Gap',
                    title="Performance Gap by District",
                    color='Performance Gap',
                    color_continuous_scale='RdYlGn',
                    text='Performance Gap'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
        
        # Category Performance Analysis
        if 'category_insights' in portfolio_data:
            st.subheader("🍽️ Performance Analysis by Category")
            
            category_data = portfolio_data['category_insights']
            df_category = pd.DataFrame(category_data).T.reset_index()
            df_category.columns = ['Category', 'Avg Performance', 'Performance Gap']
            df_category = df_category.sort_values('Avg Performance', ascending=False)
            
            fig = px.scatter(
                df_category,
                x='Avg Performance',
                y='Performance Gap',
                size=[50] * len(df_category),  # Fixed size for all points
                color='Category',
                title="Category Performance vs Gap Analysis",
                hover_data=['Category'],
                labels={'Avg Performance': 'Average Performance Score', 'Performance Gap': 'Performance Gap'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_vline(x=df_category['Avg Performance'].mean(), line_dash="dash", line_color="gray", opacity=0.7)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Optimization Candidates
        st.subheader("⚠️ Branches Needing Optimization")
        opt_candidates = portfolio_data['optimization_candidates']
        
        if opt_candidates:
            df_opt = pd.DataFrame(opt_candidates)
            df_opt['Priority'] = ['High' if gap < -0.05 else 'Medium' if gap < -0.02 else 'Low' for gap in df_opt['performance_gap']]
            
            # Color code by priority
            def highlight_priority(row):
                if row['Priority'] == 'High':
                    return ['background-color: #f8d7da'] * len(row)
                elif row['Priority'] == 'Medium':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return ['background-color: #d1ecf1'] * len(row)
            
            styled_df = df_opt.style.apply(highlight_priority, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.info("💡 **Recommendation**: Focus on branches with High priority gaps for immediate optimization opportunities.")
        else:
            st.success("🎉 No branches requiring immediate optimization - great portfolio performance!")
        
        # Action Items
        st.subheader("📝 Recommended Actions")
        
        if opt_candidates:
            high_priority = len([c for c in opt_candidates if c['performance_gap'] < -0.05])
            if high_priority > 0:
                st.error(f"🔴 **Urgent**: {high_priority} branches need immediate attention")
            
            medium_priority = len([c for c in opt_candidates if -0.05 <= c['performance_gap'] < -0.02])
            if medium_priority > 0:
                st.warning(f"🟡 **Medium**: {medium_priority} branches could benefit from optimization")
        
        action_items = [
            "🔍 **Investigate** underperforming locations for operational issues",
            "📊 **Benchmark** top performers to identify best practices",
            "🎯 **Focus** on districts with negative performance gaps",
            "📈 **Monitor** category performance trends for menu optimization",
            "🔄 **Review** locations with consistent underperformance for potential closure/relocation"
        ]
        
        for item in action_items:
            st.markdown(item)
    
    # Location Comparison
    elif page == "🔍 Location Comparison":
        st.header("🔍 Compare Multiple Locations")
        
        num_locations = st.slider("Number of locations to compare", 2, 5, 3)
        locations = []
        
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
            try:
                with st.spinner("Comparing locations..."):
                    comparison_results = ml_system.compare_locations(locations)
                    st.session_state['comparison_results'] = comparison_results
                    st.success("✅ Locations compared successfully!")
            except Exception as e:
                st.error(f"❌ Error comparing locations: {str(e)}")
        
        if 'comparison_results' in st.session_state:
            results = st.session_state['comparison_results']
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
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
            
            fig = px.bar(df_comparison, x='District', y='Score', color='Level', title="Location Scores Comparison", text='Score')
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
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
            all_categories = ['Fresh Juices', 'Coffee and Hot Beverages', 'Salads and Bowls', 'Japanese Cuisine', 'Italian Cuisine']
            focus_categories = st.multiselect("Focus Categories", all_categories, default=['Fresh Juices', 'Coffee and Hot Beverages'])
            
            if st.button("📋 Generate Report", type="primary"):
                with st.spinner("Generating comprehensive expansion report..."):
                    try:
                        report = ml_system.generate_expansion_report(target_branches=target_branches, focus_categories=focus_categories)
                        st.session_state['expansion_report'] = report
                        st.success("✅ Report generated successfully!")
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
                    
                    if 'next_steps' in report:
                        st.subheader("📝 Next Steps")
                        for step in report['next_steps']:
                            st.write(f"• {step}")
                    
                    if 'location_recommendations' in report:
                        st.subheader("📍 Location Recommendations")
                        for i, rec in enumerate(report['location_recommendations'], 1):
                            with st.expander(f"Location {i}: {rec['location']['district']}"):
                                st.write(f"**Coordinates:** {rec['location']['coordinates']}")
                                st.write(f"**Category:** {rec['location']['category']}")
                                st.write(f"**Score:** {rec['spatial_score']['score']:.3f}")
                                st.write(f"**Recommendation:** {rec['recommendation']}")
    
    # Footer
    st.markdown("---")
    st.markdown("🍽️ **Restaurant Location ML System** | Built with Streamlit")

if __name__ == "__main__":
    main()
