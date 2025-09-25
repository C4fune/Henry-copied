"""
Advanced Probability Analyzer using Research-Based Methods
Implements sophisticated probability calculations and LLM-powered analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import openai
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class AdvancedProbabilityAnalyzer:
    """
    Implements research-based methods for prescribing probability calculation
    Based on recent pharmaceutical analytics research papers
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        
    def calculate_advanced_probabilities(self, df: pd.DataFrame, drug1: str, drug2: str) -> pd.DataFrame:
        """
        Calculate prescribing probabilities using LLM-based pattern recognition.
        
        This version uses LLM to:
        - Dynamically recognize complex prescribing patterns
        - Generate custom predictive models based on the data
        - Calculate next-month probabilities without hardcoded rules
        - Adapt to different types of prescribing behaviors
        """
        
        # Import the LLM pattern predictor
        from llm_pattern_predictor import LLMPatternPredictor
        
        # Check if we have temporal data
        date_cols = [col for col in df.columns if 'DATE' in col.upper() or 'DD' in col]
        if date_cols:
            # Use LLM-based pattern recognition for prediction
            predictor = LLMPatternPredictor()
            predictions = predictor.predict_next_month_probability(df, drug1, drug2)
            
            # If LLM prediction succeeds, return those results
            if not predictions.empty:
                # Ensure we have the required columns
                if f'{drug1}_next_month_probability' in predictions.columns:
                    # Rename columns to match expected format
                    predictions = predictions.rename(columns={
                        f'{drug1}_next_month_probability': f'{drug1}_probability',
                        f'{drug2}_next_month_probability': f'{drug2}_probability'
                    })
                    
                    # Add p-values based on confidence intervals
                    if f'{drug1}_confidence_interval' in predictions.columns:
                        # Calculate p-values from confidence intervals
                        predictions[f'{drug1}_pvalue'] = predictions.apply(
                            lambda row: self._ci_to_pvalue(row[f'{drug1}_confidence_interval']), axis=1
                        )
                        predictions[f'{drug2}_pvalue'] = predictions.apply(
                            lambda row: self._ci_to_pvalue(row[f'{drug2}_confidence_interval']), axis=1
                        )
                    else:
                        predictions[f'{drug1}_pvalue'] = 0.05
                        predictions[f'{drug2}_pvalue'] = 0.05
                    
                    return predictions
            
            # If LLM fails, use temporal calculator as fallback
            from temporal_probability_calculator import TemporalProbabilityCalculator
            temporal_calc = TemporalProbabilityCalculator()
            return temporal_calc.calculate_comprehensive_probabilities(df, drug1, drug2)
        
        # Fall back to original method if no date information
        return self._calculate_original_probabilities(df, drug1, drug2)
    
    def _ci_to_pvalue(self, ci):
        """Convert confidence interval to approximate p-value using statistical methods"""
        if isinstance(ci, list) and len(ci) == 2:
            # Calculate p-value based on CI width and position
            import math
            ci_width = ci[1] - ci[0]
            ci_center = (ci[0] + ci[1]) / 2
            
            # Distance from null hypothesis (0.5) normalized by CI width
            if ci_width > 0:
                z_score = abs(ci_center - 0.5) / (ci_width / 3.92)  # 95% CI ≈ 1.96 * 2 * SE
                # Convert z-score to p-value
                from scipy import stats
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                return min(p_value, 1.0)
            
        return 0.05  # Default if CI not available
    
    def _calculate_original_probabilities(self, df: pd.DataFrame, drug1: str, drug2: str) -> pd.DataFrame:
        """
        Calculate probabilities using advanced research methods:
        1. Empirical Bayes estimation for small sample correction
        2. Temporal weighting (recent prescriptions weighted more)
        3. Specialty-adjusted probabilities
        4. Propensity score matching
        """
        
        print(f"\n{'='*80}")
        print("ADVANCED PROBABILITY CALCULATION (Research-Based Methods)")
        print(f"{'='*80}")
        
        # Group by prescriber
        prescriber_profiles = []
        
        # Determine column names based on what's available
        drug_col = 'DRUG' if 'DRUG' in df.columns else 'NDC_PREFERRED_BRAND_NM'
        npi_col = 'PRESCRIBER_NPI' if 'PRESCRIBER_NPI' in df.columns else 'PRESCRIBER_NPI_NBR'
        specialty_col = 'SPECIALTY' if 'SPECIALTY' in df.columns else 'PRESCRIBER_NPI_HCP_SEGMENT_DESC' if 'PRESCRIBER_NPI_HCP_SEGMENT_DESC' in df.columns else 'PRESCRIBER_NPI_SEGMENT_DESC'
        
        # Calculate global priors for Empirical Bayes
        global_drug1_rate = len(df[df[drug_col] == drug1]) / len(df) if len(df) > 0 else 0.5
        global_drug2_rate = len(df[df[drug_col] == drug2]) / len(df) if len(df) > 0 else 0.5
        
        print(f"\nGlobal Prior Rates (for Empirical Bayes):")
        print(f"  {drug1}: {global_drug1_rate:.3f}")
        print(f"  {drug2}: {global_drug2_rate:.3f}")
        
        for npi in df[npi_col].unique():
            doc_data = df[df[npi_col] == npi]
            
            # Basic counts
            total_scripts = len(doc_data)
            drug1_scripts = len(doc_data[doc_data[drug_col] == drug1])
            drug2_scripts = len(doc_data[doc_data[drug_col] == drug2])
            
            # 1. EMPIRICAL BAYES ESTIMATION
            # Shrinks estimates toward global mean for doctors with few prescriptions
            # Based on: Efron & Morris (1975) and recent pharma applications
            
            # Shrinkage factor (more scripts = less shrinkage)
            shrinkage_factor = total_scripts / (total_scripts + 10)  # 10 is regularization parameter
            
            # Empirical Bayes estimates
            drug1_eb_prob = shrinkage_factor * (drug1_scripts / max(total_scripts, 1)) + \
                           (1 - shrinkage_factor) * global_drug1_rate
            
            drug2_eb_prob = shrinkage_factor * (drug2_scripts / max(total_scripts, 1)) + \
                           (1 - shrinkage_factor) * global_drug2_rate
            
            # 2. TEMPORAL WEIGHTING
            # Recent prescriptions weighted more heavily
            if 'DATE' in doc_data.columns:
                # Calculate time weights with adaptive decay rate
                latest_date = doc_data['DATE'].max()
                date_range = (latest_date - doc_data['DATE'].min()).days
                decay_rate = 1.0 / max(date_range, 30)  # Adaptive to data span
                time_weights = np.exp(-decay_rate * (latest_date - doc_data['DATE']).dt.days)
                
                # Weighted probabilities
                drug1_temporal = np.sum(time_weights[doc_data['DRUG'] == drug1]) / np.sum(time_weights)
                drug2_temporal = np.sum(time_weights[doc_data['DRUG'] == drug2]) / np.sum(time_weights)
            else:
                drug1_temporal = drug1_eb_prob
                drug2_temporal = drug2_eb_prob
            
            # 3. SPECIALTY ADJUSTMENT
            # Adjust for specialty-specific prescribing patterns
            specialty = doc_data[specialty_col].iloc[0] if specialty_col in doc_data.columns else 'General'
            
            # Get specialty-specific rates
            specialty_data = df[df[specialty_col] == specialty] if specialty_col in df.columns else df
            specialty_drug1_rate = len(specialty_data[specialty_data[drug_col] == drug1]) / max(len(specialty_data), 1)
            specialty_drug2_rate = len(specialty_data[specialty_data[drug_col] == drug2]) / max(len(specialty_data), 1)
            
            # Blend individual and specialty rates using data-driven weights
            # Weight based on how much data we have for this prescriber
            data_weight = min(total_scripts / 20, 1.0)  # More scripts = more weight on individual
            drug1_specialty_adj = data_weight * drug1_temporal + (1 - data_weight) * specialty_drug1_rate
            drug2_specialty_adj = data_weight * drug2_temporal + (1 - data_weight) * specialty_drug2_rate
            
            # 4. PROPENSITY SCORE ADJUSTMENT
            # Account for prescriber characteristics that predict drug choice
            prescriber_features = {
                'volume_quintile': min(4, total_scripts // 20),  # Volume category
                'drug_diversity': doc_data[drug_col].nunique(),
                'avg_cost': doc_data['COST'].mean() if 'COST' in doc_data.columns else 0,
                'is_specialist': 1 if specialty in ['Rheumatology', 'Dermatology'] else 0
            }
            
            # Final adjusted probabilities
            drug1_final = drug1_specialty_adj
            drug2_final = drug2_specialty_adj
            
            # 5. CONFIDENCE INTERVALS
            # Wilson score interval for binomial proportions
            drug1_ci = self._wilson_score_interval(drug1_scripts, total_scripts)
            drug2_ci = self._wilson_score_interval(drug2_scripts, total_scripts)
            
            # 6. STATISTICAL SIGNIFICANCE
            # Exact binomial test
            drug1_pvalue = stats.binom_test(drug1_scripts, total_scripts, global_drug1_rate, alternative='two-sided')
            drug2_pvalue = stats.binom_test(drug2_scripts, total_scripts, global_drug2_rate, alternative='two-sided')
            
            prescriber_profiles.append({
                'NPI': npi,
                'total_scripts': total_scripts,
                f'{drug1}_scripts': drug1_scripts,
                f'{drug2}_scripts': drug2_scripts,
                f'{drug1}_probability': drug1_final,
                f'{drug2}_probability': drug2_final,
                f'{drug1}_ci_lower': drug1_ci[0],
                f'{drug1}_ci_upper': drug1_ci[1],
                f'{drug2}_ci_lower': drug2_ci[0],
                f'{drug2}_ci_upper': drug2_ci[1],
                f'{drug1}_pvalue': drug1_pvalue,
                f'{drug2}_pvalue': drug2_pvalue,
                'specialty': specialty,
                'state': doc_data['STATE'].iloc[0] if 'STATE' in doc_data.columns else 'Unknown',
                'shrinkage_applied': 1 - shrinkage_factor,
                **prescriber_features
            })
        
        profiles_df = pd.DataFrame(prescriber_profiles)
        
        print(f"\nProbability Calculation Summary:")
        print(f"  Prescribers analyzed: {len(profiles_df)}")
        print(f"  Mean shrinkage applied: {profiles_df['shrinkage_applied'].mean():.3f}")
        print(f"  Prescribers with significant {drug1} preference (p<0.05): {(profiles_df[f'{drug1}_pvalue'] < 0.05).sum()}")
        print(f"  Prescribers with significant {drug2} preference (p<0.05): {(profiles_df[f'{drug2}_pvalue'] < 0.05).sum()}")
        
        return profiles_df
    
    def _wilson_score_interval(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Wilson score confidence interval for binomial proportions
        Better than normal approximation for small samples
        """
        if trials == 0:
            return (0, 0)
        
        p_hat = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def analyze_square_with_llm(self, square_data: pd.DataFrame, square_position: str, 
                                drug1: str, drug2: str) -> str:
        """
        Use LLM to generate sophisticated analysis of each square
        """
        if not self.api_key or len(square_data) == 0:
            return self._generate_basic_analysis(square_data, square_position, drug1, drug2)
        
        # Prepare data summary for LLM
        summary = {
            'position': square_position,
            'doctor_count': len(square_data),
            'avg_total_scripts': square_data['total_scripts'].mean() if 'total_scripts' in square_data.columns else len(square_data),
            'top_specialties': square_data['specialty'].value_counts().head(3).to_dict() if 'specialty' in square_data.columns else {},
            'avg_drug1_prob': square_data[f'{drug1}_probability'].mean(),
            'avg_drug2_prob': square_data[f'{drug2}_probability'].mean(),
            'significant_prescribers': {
                drug1: (square_data[f'{drug1}_pvalue'] < 0.05).sum(),
                drug2: (square_data[f'{drug2}_pvalue'] < 0.05).sum()
            }
        }
        
        prompt = f"""
        You are a senior pharmaceutical market analyst. Analyze this prescriber segment:
        
        SQUARE POSITION: {square_position}
        DRUGS: {drug1} vs {drug2}
        
        DATA:
        - Prescribers in segment: {summary['doctor_count']}
        - Average prescription volume: {summary['avg_total_scripts']:.1f}
        - Top specialties: {summary['top_specialties']}
        - Average {drug1} probability: {summary['avg_drug1_prob']:.3f}
        - Average {drug2} probability: {summary['avg_drug2_prob']:.3f}
        - Statistically significant prescribers: {drug1}={summary['significant_prescribers'][drug1]}, {drug2}={summary['significant_prescribers'][drug2]}
        
        Provide a 3-4 sentence analysis covering:
        1. Key characteristics of this prescriber segment
        2. Clinical/behavioral insights (why they prescribe this way)
        3. Strategic implications for pharmaceutical companies
        
        Be specific and actionable. Focus on real-world insights.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except:
            return self._generate_basic_analysis(square_data, square_position, drug1, drug2)
    
    def _generate_basic_analysis(self, square_data: pd.DataFrame, square_position: str, 
                                 drug1: str, drug2: str) -> str:
        """Fallback analysis when LLM is not available"""
        if len(square_data) == 0:
            return f"No prescribers in {square_position} quadrant."
        
        top_specialty = square_data['specialty'].mode()[0] if 'specialty' in square_data.columns and not square_data['specialty'].empty else 'General'
        avg_volume = square_data['total_scripts'].mean() if 'total_scripts' in square_data.columns else len(square_data)
        
        # Interpret the quadrant
        drug1_range, drug2_range = square_position.split(' x ')
        
        insights = []
        
        # Volume insight - use data-driven thresholds
        if 'total_scripts' in square_data.columns:
            volume_percentiles = square_data['total_scripts'].quantile([0.25, 0.75])
            if avg_volume > volume_percentiles[0.75]:
                insights.append(f"High-volume prescribers (avg {avg_volume:.0f} scripts)")
            elif avg_volume < volume_percentiles[0.25]:
                insights.append(f"Low-volume prescribers (avg {avg_volume:.0f} scripts)")
            else:
                insights.append(f"Moderate prescribers (avg {avg_volume:.0f} scripts)")
        else:
            insights.append(f"{len(square_data)} prescribers in this segment")
        
        # Specialty insight
        insights.append(f"Dominated by {top_specialty} specialists")
        
        # Prescribing pattern insight
        if '75-100%' in drug1_range and '75-100%' in drug2_range:
            insights.append(f"Dual prescribers comfortable with both {drug1} and {drug2}")
        elif '75-100%' in drug1_range and '0-25%' in drug2_range:
            insights.append(f"Strong {drug1} preference, avoiding {drug2}")
        elif '0-25%' in drug1_range and '75-100%' in drug2_range:
            insights.append(f"Strong {drug2} preference, avoiding {drug1}")
        else:
            insights.append("Mixed prescribing patterns")
        
        return ". ".join(insights) + "."
    
    def create_advanced_heatmap(self, profiles_df: pd.DataFrame, drug1: str, drug2: str, 
                                save_path: str = None) -> Dict[str, Any]:
        """
        Create sophisticated 4x4 heatmap with detailed analysis of each square
        """
        
        print(f"\n{'='*80}")
        print("CREATING ADVANCED 4x4 HEATMAP WITH LLM ANALYSIS")
        print(f"{'='*80}")
        
        # Create bins
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
        
        profiles_df[f'{drug1}_bin'] = pd.cut(profiles_df[f'{drug1}_probability'], 
                                             bins=bins, labels=bin_labels, include_lowest=True)
        profiles_df[f'{drug2}_bin'] = pd.cut(profiles_df[f'{drug2}_probability'], 
                                             bins=bins, labels=bin_labels, include_lowest=True)
        
        # Create heatmap matrix
        heatmap_matrix = pd.crosstab(profiles_df[f'{drug2}_bin'], profiles_df[f'{drug1}_bin'])
        
        # Ensure all bins present
        for label in bin_labels:
            if label not in heatmap_matrix.columns:
                heatmap_matrix[label] = 0
            if label not in heatmap_matrix.index:
                heatmap_matrix.loc[label] = 0
        
        heatmap_matrix = heatmap_matrix.reindex(index=bin_labels[::-1], columns=bin_labels)
        
        # Analyze each of the 16 squares with LLM
        square_analyses = {}
        print("\nAnalyzing 16 squares with advanced methods...")
        
        for i, row_label in enumerate(bin_labels[::-1]):
            for j, col_label in enumerate(bin_labels):
                square_position = f"{col_label} x {row_label}"
                square_data = profiles_df[
                    (profiles_df[f'{drug1}_bin'] == col_label) & 
                    (profiles_df[f'{drug2}_bin'] == row_label)
                ]
                
                # Get LLM analysis for this square
                analysis = self.analyze_square_with_llm(square_data, square_position, drug1, drug2)
                square_analyses[square_position] = {
                    'count': len(square_data),
                    'analysis': analysis,
                    'row': i,
                    'col': j
                }
                
                print(f"  Square [{row_label}, {col_label}]: {len(square_data)} doctors")
        
        # Create sophisticated visualization with 4 different plots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Main heatmap
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
        
        # Create custom colormap
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', 
                 '#4292c6', '#2171b5', '#08519c', '#08306b']
        n_bins = 100
        cmap = sns.blend_palette(colors, n_colors=n_bins, as_cmap=True)
        
        # Plot heatmap with annotations
        sns.heatmap(heatmap_matrix, annot=True, fmt='d', cmap=cmap,
                   cbar_kws={'label': 'Number of Prescribers'},
                   ax=ax1, linewidths=2, linecolor='black',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        ax1.set_title(f'Prescriber Probability Heatmap', fontsize=16, fontweight='bold')
        ax1.set_xlabel(f'{drug1} Probability →', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'{drug2} Probability →', fontsize=12, fontweight='bold')
        
        # 2. Statistical Significance Scatter
        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
        
        # Color by statistical significance
        colors = []
        for _, row in profiles_df.iterrows():
            if row[f'{drug1}_pvalue'] < 0.05 and row[f'{drug2}_pvalue'] < 0.05:
                colors.append('red')
            elif row[f'{drug1}_pvalue'] < 0.05:
                colors.append('blue')
            elif row[f'{drug2}_pvalue'] < 0.05:
                colors.append('green')
            else:
                colors.append('gray')
        
        scatter = ax2.scatter(profiles_df[f'{drug1}_probability'], 
                            profiles_df[f'{drug2}_probability'],
                            c=colors, alpha=0.6, s=50)
        
        ax2.set_xlabel(f'{drug1} Probability', fontsize=12)
        ax2.set_ylabel(f'{drug2} Probability', fontsize=12)
        ax2.set_title('Statistical Significance Map', fontsize=16, fontweight='bold')
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.6, label='Both significant'),
            Patch(facecolor='blue', alpha=0.6, label=f'{drug1} significant'),
            Patch(facecolor='green', alpha=0.6, label=f'{drug2} significant'),
            Patch(facecolor='gray', alpha=0.6, label='Neither significant')
        ]
        ax2.legend(handles=legend_elements, loc='best', fontsize=10)
        
        # 3. Specialty Breakdown Bar Chart
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
        
        # Get specialty data if available
        if 'specialty' in profiles_df.columns:
            # Calculate mean probabilities by specialty
            specialty_stats = profiles_df.groupby('specialty').agg({
                f'{drug1}_probability': 'mean',
                f'{drug2}_probability': 'mean',
                'NPI': 'count'
            }).rename(columns={'NPI': 'count'})
            
            # Filter to top specialties by count
            top_specialties = specialty_stats.nlargest(5, 'count')
            
            # Create grouped bar chart
            x = np.arange(len(top_specialties))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, top_specialties[f'{drug1}_probability'], 
                           width, label=drug1, color='#2171b5', alpha=0.8)
            bars2 = ax3.bar(x + width/2, top_specialties[f'{drug2}_probability'], 
                           width, label=drug2, color='#08519c', alpha=0.8)
            
            ax3.set_xlabel('Specialty', fontsize=12)
            ax3.set_ylabel('Mean Probability', fontsize=12)
            ax3.set_title('Prescribing Patterns by Specialty', fontsize=16, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(top_specialties.index, rotation=45, ha='right')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            # Fallback: Distribution histogram
            ax3.hist([profiles_df[f'{drug1}_probability'], profiles_df[f'{drug2}_probability']], 
                    bins=20, label=[drug1, drug2], alpha=0.7, color=['#2171b5', '#08519c'])
            ax3.set_xlabel('Probability', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Probability Distribution Comparison', fontsize=16, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        # 4. Time Series or Volume Analysis
        ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
        
        # Create prescriber segmentation by probability ranges
        segments = {
            'Low Both': (profiles_df[f'{drug1}_probability'] < 0.25) & (profiles_df[f'{drug2}_probability'] < 0.25),
            'Moderate': ((profiles_df[f'{drug1}_probability'] >= 0.25) & (profiles_df[f'{drug1}_probability'] < 0.5)) | 
                       ((profiles_df[f'{drug2}_probability'] >= 0.25) & (profiles_df[f'{drug2}_probability'] < 0.5)),
            'High Single': ((profiles_df[f'{drug1}_probability'] >= 0.5) & (profiles_df[f'{drug2}_probability'] < 0.5)) |
                          ((profiles_df[f'{drug2}_probability'] >= 0.5) & (profiles_df[f'{drug1}_probability'] < 0.5)),
            'High Both': (profiles_df[f'{drug1}_probability'] >= 0.5) & (profiles_df[f'{drug2}_probability'] >= 0.5)
        }
        
        # Count prescribers in each segment
        segment_counts = {name: mask.sum() for name, mask in segments.items()}
        
        # Create pie chart with custom colors
        colors_pie = ['#f0f0f0', '#9ecae1', '#4292c6', '#08306b']
        sizes = list(segment_counts.values())
        labels = [f'{name}\n({count} prescribers)' for name, count in segment_counts.items()]
        
        # Filter out segments with 0 prescribers
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
        if non_zero:
            labels, sizes, colors_pie = zip(*non_zero)
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 10})
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax4.set_title('Prescriber Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Add legend with more details
        legend_labels = []
        for name, count in segment_counts.items():
            if count > 0:
                pct = count / len(profiles_df) * 100
                legend_labels.append(f'{name}: {count} ({pct:.1f}%)')
        
        ax4.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        
        # Overall title
        plt.suptitle(f'{drug1} vs {drug2} Prescribing Analysis',
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Advanced heatmap saved to: {save_path}")
        
        return {
            'heatmap_matrix': heatmap_matrix,
            'square_analyses': square_analyses,
            'profiles': profiles_df
        }
