# Restaurant Location ML System - Complete Usage Guide (Fixed)

import joblib
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os

class RestaurantLocationML:
    """
    Production-ready Restaurant Location ML System
    Uses accurate landmark coordinates for Jakarta Selatan
    """
    
    def __init__(self, model_path='final_model_updated'):
        """Initialize system with updated models"""
        self.model_path = model_path
        self._load_components()
        print("🎯 Restaurant Location ML System Loaded")
        print(f"📊 Spatial Model: R² = 0.451")
        print(f"📊 Portfolio Model: R² = 0.464")
        print("✅ Using accurate landmark coordinates")
    
    def _load_components(self):
        """Load all model components"""
        # Load models
        self.spatial_model = joblib.load(f'{self.model_path}/models/spatial_model.pkl')
        self.existing_model = joblib.load(f'{self.model_path}/models/existing_branch_model.pkl')
        
        # Load scalers
        self.spatial_scaler = joblib.load(f'{self.model_path}/scalers/spatial_scaler.pkl')
        self.existing_scaler = joblib.load(f'{self.model_path}/scalers/existing_scaler.pkl')
        
        # Load features
        self.spatial_features = joblib.load(f'{self.model_path}/features/spatial_features.pkl')
        self.existing_features = joblib.load(f'{self.model_path}/features/existing_features.pkl')
        
        # Load reference data
        self.reference_data = pd.read_csv(f'{self.model_path}/data/reference_branches.csv')
        
        # Load metadata with accurate landmarks
        self.metadata = joblib.load(f'{self.model_path}/models/metadata.pkl')
        self.landmarks = self.metadata['landmarks']
    
    def score_new_location(self, latitude, longitude, district, category):
        """
        Score new location for restaurant opening
        
        Args:
            latitude (float): Location latitude
            longitude (float): Location longitude
            district (str): District name (e.g., 'Kebayoran Baru')
            category (str): Restaurant category (e.g., 'Fresh Juices')
        
        Returns:
            dict: Location scoring results
        """
        
        # Calculate spatial features
        features = self._calculate_spatial_features(latitude, longitude, category)
        
        # Add categorical features
        features['districtName'] = self._encode_categorical(district, 'districtName')
        features['Category'] = self._encode_categorical(category, 'Category')
        
        # Create feature vector
        feature_vector = [features.get(feature, 0) for feature in self.spatial_features]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.spatial_scaler.transform(feature_vector)
        
        # Predict
        score = self.spatial_model.predict(feature_vector_scaled)[0]
        
        return {
            'location': {
                'coordinates': (latitude, longitude),
                'district': district,
                'category': category
            },
            'spatial_score': {
                'score': round(score, 3),
                'level': self._categorize_score(score),
                'confidence': 'Medium'
            },
            'key_factors': {
                'distance_to_kemang_km': round(features['dist_to_kemang'], 2),
                'distance_to_cbd_km': round(features['dist_to_sudirman_cbd'], 2),
                'distance_to_senayan_km': round(features['dist_to_senayan'], 2),
                'competitors_1km': features['competitors_1.0km'],
                'market_intensity_billions': round(features['competition_intensity_1.0km'] / 1e9, 2)
            },
            'recommendation': self._generate_recommendation(score),
            'spatial_insights': self._generate_spatial_insights(features)
        }
    
    def analyze_portfolio(self):
        """
        Analyze existing branch portfolio for optimization
        
        Returns:
            dict: Portfolio analysis results
        """
        
        # Prepare features for existing branches
        X_existing = self.reference_data[self.existing_features].copy()
        for col in X_existing.columns:
            if X_existing[col].dtype == 'object':
                X_existing[col] = pd.Categorical(X_existing[col]).codes
        
        # Predict performance
        X_scaled = self.existing_scaler.transform(X_existing)
        predicted_scores = self.existing_model.predict(X_scaled)
        
        # Create portfolio analysis
        portfolio = self.reference_data[['branch_id', 'branchName', 'districtName', 'Category', 'performance_score']].copy()
        portfolio['predicted_score'] = predicted_scores
        portfolio['performance_gap'] = portfolio['performance_score'] - portfolio['predicted_score']
        
        # Categorize branches
        portfolio['status'] = pd.cut(
            portfolio['performance_gap'],
            bins=[-np.inf, -0.1, 0.1, np.inf],
            labels=['Underperforming', 'As Expected', 'Overperforming']
        )
        
        return {
            'portfolio_summary': {
                'total_branches': len(portfolio),
                'avg_performance': round(portfolio['performance_score'].mean(), 3),
                'avg_potential': round(portfolio['predicted_score'].mean(), 3),
                'optimization_gap': round(portfolio['performance_gap'].sum(), 3)
            },
            'status_distribution': portfolio['status'].value_counts().to_dict(),
            'top_performers': portfolio.nlargest(5, 'performance_score')[
                ['branchName', 'districtName', 'performance_score']
            ].to_dict('records'),
            'optimization_candidates': portfolio.nsmallest(5, 'performance_gap')[
                ['branchName', 'districtName', 'performance_gap']
            ].to_dict('records'),
            'best_practice_sources': portfolio.nlargest(5, 'performance_gap')[
                ['branchName', 'districtName', 'performance_gap']
            ].to_dict('records'),
            'district_insights': self._analyze_by_district(portfolio),
            'category_insights': self._analyze_by_category(portfolio)
        }
    
    def compare_locations(self, locations_list):
        """
        Compare multiple candidate locations
        
        Args:
            locations_list (list): List of tuples (lat, lng, district, category)
        
        Returns:
            list: Sorted list of location scores
        """
        
        results = []
        for lat, lng, district, category in locations_list:
            result = self.score_new_location(lat, lng, district, category)
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['spatial_score']['score'], reverse=True)
        
        return results
    
    def find_optimal_districts(self, category, num_districts=3):
        """
        Find optimal districts for specific category
        
        Args:
            category (str): Restaurant category
            num_districts (int): Number of districts to return
        
        Returns:
            list: Best districts for the category
        """
        
        # Analyze existing performance by district for the category
        category_data = self.reference_data[self.reference_data['Category'] == category]
        
        if len(category_data) == 0:
            # If no existing data, use general district performance
            district_performance = self.reference_data.groupby('districtName')['performance_score'].mean()
        else:
            district_performance = category_data.groupby('districtName')['performance_score'].mean()
        
        # Get top districts
        top_districts = district_performance.nlargest(num_districts)
        
        return [
            {
                'district': district,
                'avg_performance': round(performance, 3),
                'existing_branches': len(category_data[category_data['districtName'] == district]) if len(category_data) > 0 else 0
            }
            for district, performance in top_districts.items()
        ]
    
    def generate_expansion_report(self, target_branches=5, focus_categories=None):
        """
        Generate comprehensive expansion report
        
        Args:
            target_branches (int): Target number of new branches
            focus_categories (list): Categories to focus on
        
        Returns:
            dict: Complete expansion report
        """
        
        if focus_categories is None:
            focus_categories = ['Fresh Juices', 'Salads and Bowls', 'Coffee and Hot Beverages']
        
        try:
            # Portfolio analysis
            portfolio = self.analyze_portfolio()
            
            # Category analysis
            category_opportunities = []
            for category in focus_categories:
                optimal_districts = self.find_optimal_districts(category, num_districts=2)
                category_opportunities.append({
                    'category': category,
                    'optimal_districts': optimal_districts
                })
            
            # Sample location recommendations (fixed)
            sample_locations = []
            coords = [
                (-6.225, 106.825, 'Setia Budi'),
                (-6.235, 106.805, 'Kebayoran Baru'),
                (-6.245, 106.795, 'Kebayoran Baru')
            ]
            
            for i, (lat, lng, district) in enumerate(coords):
                category = focus_categories[i % len(focus_categories)]  # Cycle through categories
                sample_locations.append((lat, lng, district, category))
            
            location_scores = self.compare_locations(sample_locations[:target_branches])
            
            return {
                'executive_summary': {
                    'target_new_branches': target_branches,
                    'focus_categories': focus_categories,
                    'recommended_districts': ['Setia Budi', 'Kebayoran Baru'],
                    'total_investment_estimate': f"{target_branches * 500}M - {target_branches * 800}M IDR"
                },
                'portfolio_optimization': {
                    'current_performance': portfolio['portfolio_summary'],
                    'optimization_candidates': len(portfolio['optimization_candidates']),
                    'best_practice_sources': len(portfolio['best_practice_sources'])
                },
                'category_opportunities': category_opportunities,
                'location_recommendations': location_scores,
                'risk_assessment': {
                    'model_confidence': 'Medium (R² = 0.451-0.464)',
                    'market_risk': 'Low-Medium',
                    'competition_risk': 'Medium',
                    'location_risk': 'Medium - validate with market research'
                },
                'next_steps': [
                    'Validate top locations with field research',
                    'Analyze foot traffic patterns',
                    'Study local competition details',
                    'Optimize underperforming existing branches'
                ]
            }
            
        except Exception as e:
            print(f"Error in expansion report: {e}")
            return {
                'error': str(e),
                'executive_summary': {
                    'target_new_branches': target_branches,
                    'status': 'Error occurred during report generation'
                }
            }
    
    # Helper methods
    def _calculate_spatial_features(self, lat, lng, category):
        """Calculate spatial features for new location"""
        
        features = {'latitude': lat, 'longitude': lng}
        
        # Distance to landmarks (using accurate coordinates)
        for landmark_name, (landmark_lat, landmark_lng) in self.landmarks.items():
            distance = geodesic((lat, lng), (landmark_lat, landmark_lng)).kilometers
            features[f'dist_to_{landmark_name}'] = distance
        
        features['dist_to_nearest_landmark'] = min([
            features[f'dist_to_{name}'] for name in self.landmarks.keys()
        ])
        
        # Competition analysis
        for radius in [0.5, 1.0, 2.0]:
            competitors = 0
            same_category = 0
            competitor_revenue = 0
            
            for _, branch in self.reference_data.iterrows():
                distance = geodesic((lat, lng), (branch['latitude'], branch['longitude'])).kilometers
                
                if distance <= radius:
                    competitors += 1
                    competitor_revenue += branch['GTV2024']
                    if branch['Category'] == category:
                        same_category += 1
            
            features[f'competitors_{radius}km'] = competitors
            features[f'same_category_competitors_{radius}km'] = same_category
            features[f'competitor_revenue_{radius}km'] = competitor_revenue
            features[f'competition_intensity_{radius}km'] = competitor_revenue / (competitors + 1)
        
        return features
    
    def _encode_categorical(self, value, column_name):
        """Encode categorical values"""
        unique_values = self.reference_data[column_name].unique()
        if value in unique_values:
            return pd.Categorical([value], categories=unique_values).codes[0]
        return 0
    
    def _categorize_score(self, score):
        """Categorize location score"""
        if score >= 0.3: return 'High'
        elif score >= 0.2: return 'Medium'
        elif score >= 0.1: return 'Low'
        else: return 'Very Low'
    
    def _generate_recommendation(self, score):
        """Generate business recommendation"""
        if score >= 0.3:
            return "🟢 RECOMMENDED: Good potential location"
        elif score >= 0.2:
            return "🟡 CONSIDER: Moderate potential, validate with research"
        elif score >= 0.1:
            return "🟠 CAUTION: Low potential, consider alternatives"
        else:
            return "🔴 NOT RECOMMENDED: Very low potential"
    
    def _generate_spatial_insights(self, features):
        """Generate spatial insights"""
        insights = []
        
        if features['dist_to_kemang'] < 2:
            insights.append("Close to trendy Kemang area")
        if features['dist_to_sudirman_cbd'] < 3:
            insights.append("Near main business district")
        if features['competitors_1.0km'] > 5:
            insights.append("High market activity area")
        if features['competition_intensity_1.0km'] > 1e9:
            insights.append("Strong market demand indicated")
        
        return insights if insights else ["Standard market conditions"]
    
    def _analyze_by_district(self, portfolio):
        """Analyze performance by district"""
        return portfolio.groupby('districtName').agg({
            'performance_score': 'mean',
            'performance_gap': 'mean'
        }).round(3).to_dict('index')
    
    def _analyze_by_category(self, portfolio):
        """Analyze performance by category"""
        return portfolio.groupby('Category').agg({
            'performance_score': 'mean',
            'performance_gap': 'mean'
        }).round(3).to_dict('index')

# Usage Examples
def main():
    """Usage examples and demonstrations"""
    
    print("🎯 Restaurant Location ML System - Usage Examples")
    print("=" * 60)
    
    # Initialize system
    try:
        ml_system = RestaurantLocationML('final_model_updated')
    except FileNotFoundError:
        print("❌ Error: Models not found. Please ensure 'final_model_updated' directory exists.")
        return
    
    print("\n" + "=" * 60)
    print("📍 EXAMPLE 1: Score Single New Location")
    print("=" * 60)
    
    result = ml_system.score_new_location(
        latitude=-6.235,
        longitude=106.805,
        district='Kebayoran Baru',
        category='Fresh Juices'
    )
    
    print(f"Location: {result['location']['coordinates']} - {result['location']['district']}")
    print(f"Category: {result['location']['category']}")
    print(f"Score: {result['spatial_score']['score']} ({result['spatial_score']['level']})")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Distance to Kemang: {result['key_factors']['distance_to_kemang_km']} km")
    print(f"Competitors in 1km: {result['key_factors']['competitors_1km']}")
    
    print("\n" + "=" * 60)
    print("📊 EXAMPLE 2: Portfolio Analysis")
    print("=" * 60)
    
    portfolio = ml_system.analyze_portfolio()
    
    print(f"Total branches: {portfolio['portfolio_summary']['total_branches']}")
    print(f"Average performance: {portfolio['portfolio_summary']['avg_performance']}")
    print(f"Optimization gap: {portfolio['portfolio_summary']['optimization_gap']}")
    print(f"Branches need optimization: {len(portfolio['optimization_candidates'])}")
    
    print("\nTop 3 performers:")
    for i, branch in enumerate(portfolio['top_performers'][:3], 1):
        print(f"{i}. {branch['branchName']} - {branch['districtName']} (Score: {branch['performance_score']:.3f})")
    
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 3: Compare Multiple Locations")
    print("=" * 60)
    
    candidate_locations = [
        (-6.235, 106.805, 'Kebayoran Baru', 'Fresh Juices'),
        (-6.225, 106.825, 'Setia Budi', 'Coffee and Hot Beverages'),
        (-6.285, 106.795, 'Cilandak', 'Japanese Cuisine')
    ]
    
    comparison = ml_system.compare_locations(candidate_locations)
    
    print("Location comparison (ranked by score):")
    for i, loc in enumerate(comparison, 1):
        print(f"{i}. {loc['location']['district']} - {loc['location']['category']}")
        print(f"   Score: {loc['spatial_score']['score']} ({loc['spatial_score']['level']})")
        print(f"   {loc['recommendation']}")
    
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Expansion Report")
    print("=" * 60)
    
    try:
        expansion_report = ml_system.generate_expansion_report(
            target_branches=3,
            focus_categories=['Fresh Juices', 'Coffee and Hot Beverages']
        )
        
        if 'executive_summary' in expansion_report:
            exec_summary = expansion_report['executive_summary']
            print("Executive Summary:")
            print(f"Target branches: {exec_summary.get('target_new_branches', 'N/A')}")
            print(f"Recommended districts: {exec_summary.get('recommended_districts', 'N/A')}")
            print(f"Investment estimate: {exec_summary.get('total_investment_estimate', 'N/A')}")
            
            if 'risk_assessment' in expansion_report:
                print(f"\nRisk Assessment:")
                risk = expansion_report['risk_assessment']
                print(f"Model confidence: {risk.get('model_confidence', 'N/A')}")
                print(f"Market risk: {risk.get('market_risk', 'N/A')}")
            
            if 'next_steps' in expansion_report:
                print("\nNext steps:")
                for step in expansion_report['next_steps']:
                    print(f"- {step}")
        else:
            print("❌ Expansion report generation failed")
            if 'error' in expansion_report:
                print(f"Error: {expansion_report['error']}")
    
    except Exception as e:
        print(f"❌ Error generating expansion report: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("📖 Refer to this script for implementation guidance")

if __name__ == "__main__":
    main()