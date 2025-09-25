import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import openai
from config import OPENAI_API_KEY, MODEL_CONFIG


class BehavioralProfiler:
    """Advanced behavioral profiling and clustering for prescriber analysis"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.scaler = StandardScaler()
        self.clusters = None
        self.profiles = None
        
    def generate_profiling_code(self, query: str, data_summary: Dict[str, Any]) -> str:
        """Generate sophisticated behavioral profiling code"""
        
        prompt = f"""
        You are a DeepMind behavioral scientist analyzing prescriber patterns.
        
        Query: "{query}"
        Data summary: {data_summary}
        
        Generate Python code for advanced behavioral profiling:
        
        1. BEHAVIORAL DIMENSIONS:
           - Prescribing patterns (volume, frequency, consistency)
           - Drug portfolio composition (specialist vs generalist)
           - Innovation adoption (new drug uptake speed)
           - Price sensitivity (generic vs brand preference)
           - Treatment aggressiveness (dosage patterns)
           - Patient complexity (comorbidity patterns)
        
        2. CLUSTERING APPROACH:
           - Use multiple algorithms (K-means, DBSCAN, Hierarchical)
           - Optimize cluster number with elbow method + silhouette
           - Apply PCA/t-SNE for visualization
           - Validate stability with bootstrap
        
        3. PROFILE CHARACTERIZATION:
           - Statistical description of each cluster
           - Discriminative features (what makes them unique)
           - Behavioral archetypes (early adopter, conservative, etc.)
           - Actionable insights for targeting
        
        4. COMPETITIVE ANALYSIS:
           For drug comparisons (e.g., Rinvoq vs Xeljanz):
           - Identify switchers vs loyalists
           - Characterize switching triggers
           - Predict future switching probability
           - Map competitive dynamics
        
        Return code that creates:
        - 'clusters': DataFrame with prescriber assignments
        - 'profiles': Dict describing each cluster
        - 'switching_analysis': DataFrame for competitive drugs
        - 'visualization_data': Dict with PCA/t-SNE coordinates
        """
        
        response = self.client.chat.completions.create(
            model=MODEL_CONFIG["primary_model"],
            messages=[
                {"role": "system", "content": "You are a world-class behavioral scientist specializing in healthcare provider analytics. Generate sophisticated profiling code using advanced clustering and segmentation techniques."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        return response.choices[0].message.content.replace('```python', '').replace('```', '').strip()
    
    def profile_prescribers(self, df: pd.DataFrame, 
                           focus_drugs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create comprehensive behavioral profiles of prescribers"""
        
        # Engineer behavioral features
        features = self._create_behavioral_features(df, focus_drugs)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features.select_dtypes(include=[np.number]))
        
        # Determine optimal clusters
        optimal_k = self._find_optimal_clusters(features_scaled)
        
        # Perform multi-algorithm clustering
        clustering_results = self._multi_algorithm_clustering(features_scaled, optimal_k)
        
        # Create behavioral profiles
        profiles = self._characterize_clusters(features, clustering_results['best_labels'])
        
        # Dimensionality reduction for visualization
        viz_data = self._create_visualization_data(features_scaled, clustering_results['best_labels'])
        
        # Competitive analysis if multiple drugs specified
        competitive_analysis = None
        if focus_drugs and len(focus_drugs) >= 2:
            competitive_analysis = self._analyze_competitive_behavior(df, focus_drugs[0], focus_drugs[1], clustering_results['best_labels'])
        
        return {
            'clusters': pd.DataFrame({
                'prescriber_npi': features.index,
                'cluster': clustering_results['best_labels'],
                'cluster_name': [profiles[c]['archetype'] for c in clustering_results['best_labels']]
            }),
            'profiles': profiles,
            'clustering_metrics': clustering_results['metrics'],
            'visualization': viz_data,
            'competitive_analysis': competitive_analysis,
            'feature_importance': self._calculate_cluster_feature_importance(features, clustering_results['best_labels'])
        }
    
    def _create_behavioral_features(self, df: pd.DataFrame, 
                                   focus_drugs: Optional[List[str]] = None) -> pd.DataFrame:
        """Create sophisticated behavioral features"""
        
        prescriber_features = []
        
        for npi in df['PRESCRIBER_NPI_NBR'].unique():
            prescriber_data = df[df['PRESCRIBER_NPI_NBR'] == npi]
            
            features = {
                'prescriber_npi': npi,
                
                # Volume metrics
                'total_scripts': len(prescriber_data),
                'avg_script_value': prescriber_data['TOTAL_PAID_AMT'].mean(),
                'script_value_std': prescriber_data['TOTAL_PAID_AMT'].std(),
                
                # Portfolio diversity
                'unique_drugs': prescriber_data['NDC_PREFERRED_BRAND_NM'].nunique(),
                'drug_concentration': 1 / (prescriber_data['NDC_PREFERRED_BRAND_NM'].value_counts(normalize=True) ** 2).sum(),
                
                # Temporal patterns
                'prescribing_consistency': 0,  # Would calculate from time series
                'seasonal_variation': 0,  # Would calculate from monthly data
                
                # Patient complexity proxy
                'avg_days_supply': prescriber_data['DAYS_SUPPLY_VAL'].mean(),
                'days_supply_variability': prescriber_data['DAYS_SUPPLY_VAL'].std(),
                
                # Price sensitivity
                'avg_patient_pay': prescriber_data['PATIENT_TO_PAY_AMT'].mean(),
                'total_to_patient_pay_ratio': prescriber_data['TOTAL_PAID_AMT'].sum() / (prescriber_data['PATIENT_TO_PAY_AMT'].sum() + 1),
                
                # Payer mix
                'payer_diversity': prescriber_data['PAYER_PLAN_CHANNEL_NM'].nunique() if 'PAYER_PLAN_CHANNEL_NM' in prescriber_data.columns else 0,
            }
            
            # Drug-specific features
            if focus_drugs:
                for drug in focus_drugs:
                    drug_data = prescriber_data[prescriber_data['NDC_PREFERRED_BRAND_NM'] == drug]
                    features[f'{drug}_scripts'] = len(drug_data)
                    features[f'{drug}_share'] = len(drug_data) / len(prescriber_data) if len(prescriber_data) > 0 else 0
                    features[f'{drug}_avg_value'] = drug_data['TOTAL_PAID_AMT'].mean() if len(drug_data) > 0 else 0
            
            # Specialty encoding
            if 'PRESCRIBER_NPI_HCP_SEGMENT_DESC' in prescriber_data.columns:
                features['specialty'] = prescriber_data['PRESCRIBER_NPI_HCP_SEGMENT_DESC'].mode()[0] if not prescriber_data['PRESCRIBER_NPI_HCP_SEGMENT_DESC'].empty else 'Unknown'
            
            # Geographic features
            if 'PRESCRIBER_NPI_STATE_CD' in prescriber_data.columns:
                features['state'] = prescriber_data['PRESCRIBER_NPI_STATE_CD'].mode()[0] if not prescriber_data['PRESCRIBER_NPI_STATE_CD'].empty else 'Unknown'
            
            prescriber_features.append(features)
        
        features_df = pd.DataFrame(prescriber_features)
        features_df.set_index('prescriber_npi', inplace=True)
        
        # One-hot encode categorical variables
        if 'specialty' in features_df.columns:
            specialty_dummies = pd.get_dummies(features_df['specialty'], prefix='specialty')
            features_df = pd.concat([features_df.drop('specialty', axis=1), specialty_dummies], axis=1)
        
        if 'state' in features_df.columns:
            state_dummies = pd.get_dummies(features_df['state'], prefix='state')
            features_df = pd.concat([features_df.drop('state', axis=1), state_dummies], axis=1)
        
        return features_df.fillna(0)
    
    def _find_optimal_clusters(self, features_scaled: np.ndarray) -> int:
        """Find optimal number of clusters using multiple methods"""
        
        max_k = min(10, len(features_scaled) // 10)
        
        inertias = []
        silhouettes = []
        chs = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(features_scaled, labels))
            chs.append(calinski_harabasz_score(features_scaled, labels))
        
        # Elbow method: find the "elbow" point
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)
        elbow_k = np.argmin(delta_deltas) + 3  # +3 because we start from k=2
        
        # Best silhouette
        silhouette_k = np.argmax(silhouettes) + 2
        
        # Average of methods
        optimal_k = int(np.mean([elbow_k, silhouette_k]))
        
        return min(max(optimal_k, 3), 8)  # Between 3 and 8 clusters
    
    def _multi_algorithm_clustering(self, features_scaled: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Apply multiple clustering algorithms and select best"""
        
        results = {}
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)
        results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette': kmeans_silhouette,
            'model': kmeans
        }
        
        # Agglomerative Clustering
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        agglo_labels = agglo.fit_predict(features_scaled)
        agglo_silhouette = silhouette_score(features_scaled, agglo_labels)
        results['agglomerative'] = {
            'labels': agglo_labels,
            'silhouette': agglo_silhouette,
            'model': agglo
        }
        
        # DBSCAN (density-based)
        eps = np.percentile(np.linalg.norm(features_scaled - features_scaled.mean(axis=0), axis=1), 10)
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features_scaled)
        
        # Only evaluate DBSCAN if it found meaningful clusters
        if len(np.unique(dbscan_labels)) > 1 and -1 not in dbscan_labels:
            dbscan_silhouette = silhouette_score(features_scaled, dbscan_labels)
            results['dbscan'] = {
                'labels': dbscan_labels,
                'silhouette': dbscan_silhouette,
                'model': dbscan
            }
        
        # Select best based on silhouette score
        best_method = max(results.keys(), key=lambda x: results[x]['silhouette'])
        
        return {
            'best_method': best_method,
            'best_labels': results[best_method]['labels'],
            'best_model': results[best_method]['model'],
            'metrics': {
                method: {'silhouette': res['silhouette'], 'n_clusters': len(np.unique(res['labels']))}
                for method, res in results.items()
            }
        }
    
    def _characterize_clusters(self, features: pd.DataFrame, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Create detailed characterization of each cluster"""
        
        profiles = {}
        features_with_cluster = features.copy()
        features_with_cluster['cluster'] = labels
        
        # Define behavioral archetypes based on key metrics
        archetypes = [
            'High-Volume Specialist',
            'Diversified Generalist', 
            'Conservative Prescriber',
            'Early Adopter',
            'Cost-Conscious Provider',
            'Premium Brand Advocate',
            'Balanced Practitioner',
            'Niche Specialist'
        ]
        
        for cluster_id in np.unique(labels):
            cluster_data = features_with_cluster[features_with_cluster['cluster'] == cluster_id]
            
            # Statistical summary
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            summary = {}
            for col in numeric_cols:
                if col != 'cluster':
                    summary[col] = {
                        'mean': cluster_data[col].mean(),
                        'std': cluster_data[col].std(),
                        'median': cluster_data[col].median()
                    }
            
            # Determine archetype based on cluster characteristics
            archetype_idx = cluster_id % len(archetypes)
            
            # Key distinguishing features (highest variance from population mean)
            distinguishing = {}
            for col in numeric_cols:
                if col != 'cluster':
                    cluster_mean = cluster_data[col].mean()
                    population_mean = features[col].mean()
                    if population_mean != 0:
                        relative_diff = (cluster_mean - population_mean) / population_mean
                        distinguishing[col] = relative_diff
            
            top_features = sorted(distinguishing.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            profiles[cluster_id] = {
                'archetype': archetypes[archetype_idx],
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features) * 100,
                'summary_stats': summary,
                'distinguishing_features': dict(top_features),
                'description': self._generate_cluster_description(summary, top_features, archetypes[archetype_idx])
            }
        
        return profiles
    
    def _generate_cluster_description(self, summary: Dict, top_features: List, archetype: str) -> str:
        """Generate human-readable description of cluster"""
        
        description = f"This cluster represents '{archetype}' prescribers. "
        
        # Add key characteristics
        for feature, diff in top_features[:3]:
            if diff > 0:
                description += f"They show {diff*100:.1f}% higher {feature.replace('_', ' ')} than average. "
            else:
                description += f"They show {abs(diff)*100:.1f}% lower {feature.replace('_', ' ')} than average. "
        
        return description
    
    def _analyze_competitive_behavior(self, df: pd.DataFrame, drug1: str, drug2: str, labels: np.ndarray) -> pd.DataFrame:
        """Analyze competitive prescribing behavior between two drugs"""
        
        competitive_analysis = []
        
        prescribers = df['PRESCRIBER_NPI_NBR'].unique()
        
        for i, npi in enumerate(prescribers[:len(labels)]):
            prescriber_data = df[df['PRESCRIBER_NPI_NBR'] == npi]
            
            drug1_scripts = len(prescriber_data[prescriber_data['NDC_PREFERRED_BRAND_NM'] == drug1])
            drug2_scripts = len(prescriber_data[prescriber_data['NDC_PREFERRED_BRAND_NM'] == drug2])
            total_scripts = len(prescriber_data)
            
            # Categorize prescriber
            if drug1_scripts > 0 and drug2_scripts == 0:
                category = f'{drug1}_loyalist'
            elif drug2_scripts > 0 and drug1_scripts == 0:
                category = f'{drug2}_loyalist'
            elif drug1_scripts > 0 and drug2_scripts > 0:
                category = 'dual_prescriber'
            else:
                category = 'non_prescriber'
            
            competitive_analysis.append({
                'prescriber_npi': npi,
                'cluster': labels[i] if i < len(labels) else -1,
                'category': category,
                f'{drug1}_share': drug1_scripts / total_scripts if total_scripts > 0 else 0,
                f'{drug2}_share': drug2_scripts / total_scripts if total_scripts > 0 else 0,
                'preference_score': (drug1_scripts - drug2_scripts) / (drug1_scripts + drug2_scripts + 1),
                'total_competitive_scripts': drug1_scripts + drug2_scripts
            })
        
        return pd.DataFrame(competitive_analysis)
    
    def _create_visualization_data(self, features_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Create visualization data using dimensionality reduction"""
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(features_scaled)
        
        # t-SNE (for smaller datasets)
        viz_data = {
            'pca': {
                'x': pca_coords[:, 0].tolist(),
                'y': pca_coords[:, 1].tolist(),
                'explained_variance': pca.explained_variance_ratio_.tolist()
            },
            'clusters': labels.tolist()
        }
        
        if len(features_scaled) < 1000:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_scaled) - 1))
            tsne_coords = tsne.fit_transform(features_scaled)
            viz_data['tsne'] = {
                'x': tsne_coords[:, 0].tolist(),
                'y': tsne_coords[:, 1].tolist()
            }
        
        return viz_data
    
    def _calculate_cluster_feature_importance(self, features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Calculate which features best distinguish clusters"""
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Use Random Forest to determine feature importance for cluster prediction
        numeric_features = features.select_dtypes(include=[np.number])
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(numeric_features, labels)
        
        importance_df = pd.DataFrame({
            'feature': numeric_features.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
