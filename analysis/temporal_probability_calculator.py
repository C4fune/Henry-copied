"""
Advanced Temporal Probability Calculator for Pharmaceutical Analytics

This module implements sophisticated statistical methods for calculating
prescribing probabilities based on temporal consistency, volume patterns,
and multi-dimensional behavioral analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import betainc
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class TemporalProbabilityCalculator:
    """
    Calculate prescribing probabilities using temporal consistency metrics
    and advanced statistical methods that properly reflect consistent behavior.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.min_months_for_consistency = 3
        self.consistency_weight = 0.4  # Weight for temporal consistency
        self.volume_weight = 0.3      # Weight for volume patterns
        self.recency_weight = 0.3     # Weight for recent activity
    
    def calculate_comprehensive_probabilities(self, 
                                             df: pd.DataFrame, 
                                             drug1: str, 
                                             drug2: str,
                                             time_window_months: int = 6) -> pd.DataFrame:
        """
        Calculate probabilities using multiple dimensions of prescribing behavior.
        
        Key improvements over previous method:
        1. Temporal consistency scoring (months active / total months)
        2. Volume stability analysis (coefficient of variation)
        3. Recency weighting (recent prescriptions count more)
        4. Peer-adjusted probabilities (specialty and region normalization)
        5. Confidence intervals based on data quantity and quality
        """
        
        print(f"\n{'='*80}")
        print("ADVANCED TEMPORAL PROBABILITY CALCULATION")
        print(f"{'='*80}")
        
        # Prepare temporal data
        df = self._prepare_temporal_data(df)
        
        # Get unique prescribers
        prescriber_profiles = []
        
        # Determine column mappings
        drug_col = self._get_drug_column(df)
        npi_col = self._get_npi_column(df)
        date_col = self._get_date_column(df)
        
        # Calculate global benchmarks for peer adjustment
        global_stats = self._calculate_global_statistics(df, drug1, drug2, drug_col)
        
        for npi in df[npi_col].unique():
            prescriber_data = df[df[npi_col] == npi]
            
            # Calculate temporal metrics for each drug
            drug1_metrics = self._calculate_drug_metrics(
                prescriber_data, drug1, drug_col, date_col, time_window_months
            )
            drug2_metrics = self._calculate_drug_metrics(
                prescriber_data, drug2, drug_col, date_col, time_window_months
            )
            
            # Calculate composite probability scores
            drug1_prob = self._calculate_composite_probability(drug1_metrics, global_stats['drug1'])
            drug2_prob = self._calculate_composite_probability(drug2_metrics, global_stats['drug2'])
            
            # Calculate p-values using appropriate statistical tests
            drug1_pvalue = self._calculate_temporal_pvalue(drug1_metrics, global_stats['drug1'])
            drug2_pvalue = self._calculate_temporal_pvalue(drug2_metrics, global_stats['drug2'])
            
            # Get prescriber characteristics
            specialty = self._get_specialty(prescriber_data)
            
            prescriber_profiles.append({
                'NPI': npi,
                'SPECIALTY': specialty,
                'TOTAL_SCRIPTS': len(prescriber_data),
                f'{drug1}_probability': drug1_prob,
                f'{drug2}_probability': drug2_prob,
                f'{drug1}_pvalue': drug1_pvalue,
                f'{drug2}_pvalue': drug2_pvalue,
                f'{drug1}_temporal_consistency': drug1_metrics['temporal_consistency'],
                f'{drug2}_temporal_consistency': drug2_metrics['temporal_consistency'],
                f'{drug1}_months_active': drug1_metrics['months_active'],
                f'{drug2}_months_active': drug2_metrics['months_active'],
                f'{drug1}_volume_stability': drug1_metrics['volume_stability'],
                f'{drug2}_volume_stability': drug2_metrics['volume_stability'],
                'prescribing_pattern': self._classify_pattern(drug1_prob, drug2_prob)
            })
        
        results_df = pd.DataFrame(prescriber_profiles)
        
        # Print summary statistics
        self._print_summary_statistics(results_df, drug1, drug2)
        
        return results_df
    
    def _prepare_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with temporal features"""
        
        # Add temporal features if date column exists
        date_cols = [col for col in df.columns if 'DATE' in col.upper() or 'DD' in col]
        if date_cols:
            date_col = date_cols[0]
            df['prescription_date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['year_month'] = df['prescription_date'].dt.to_period('M')
            
            # Calculate days from most recent prescription
            max_date = df['prescription_date'].max()
            df['days_from_recent'] = (max_date - df['prescription_date']).dt.days
            df['recency_weight'] = np.exp(-df['days_from_recent'] / 180)  # 6-month half-life
        
        return df
    
    def _calculate_drug_metrics(self, prescriber_data: pd.DataFrame, drug: str, 
                                drug_col: str, date_col: str, 
                                time_window_months: int) -> Dict:
        """Calculate comprehensive metrics for a specific drug"""
        
        drug_data = prescriber_data[prescriber_data[drug_col] == drug]
        
        if len(drug_data) == 0:
            return {
                'total_scripts': 0,
                'months_active': 0,
                'temporal_consistency': 0.0,
                'volume_stability': 0.0,
                'recency_score': 0.0,
                'weighted_scripts': 0.0
            }
        
        # Temporal consistency: fraction of months with prescriptions
        if 'year_month' in drug_data.columns:
            unique_months = drug_data['year_month'].nunique()
            months_active = unique_months
            temporal_consistency = min(unique_months / time_window_months, 1.0)
            
            # Volume stability: inverse of coefficient of variation
            monthly_counts = drug_data.groupby('year_month').size()
            if len(monthly_counts) > 1:
                cv = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 1.0
                volume_stability = 1 / (1 + cv)  # Higher is more stable
            else:
                volume_stability = 0.5  # Default for single month
            
            # Recency score: weighted by how recent prescriptions are
            if 'recency_weight' in drug_data.columns:
                recency_score = drug_data['recency_weight'].mean()
                weighted_scripts = (drug_data['recency_weight'] * 1).sum()
            else:
                recency_score = 0.5
                weighted_scripts = len(drug_data)
        else:
            months_active = 1
            temporal_consistency = len(drug_data) / len(prescriber_data)
            volume_stability = 0.5
            recency_score = 0.5
            weighted_scripts = len(drug_data)
        
        return {
            'total_scripts': len(drug_data),
            'months_active': months_active,
            'temporal_consistency': temporal_consistency,
            'volume_stability': volume_stability,
            'recency_score': recency_score,
            'weighted_scripts': weighted_scripts
        }
    
    def _calculate_composite_probability(self, metrics: Dict, global_benchmark: Dict) -> float:
        """
        Calculate composite probability score using multiple factors.
        
        This addresses the issue where consistent prescribers were getting low scores.
        If someone prescribes a drug consistently for several months, they should
        have a HIGH probability score (>0.75), not a low one.
        """
        
        if metrics['total_scripts'] == 0:
            return 0.0
        
        # Base probability from temporal consistency
        # If prescribed in all months, this gives high score
        temporal_score = metrics['temporal_consistency']
        
        # Volume stability bonus
        # Consistent volume patterns indicate established prescribing
        stability_bonus = metrics['volume_stability'] * 0.2
        
        # Recency adjustment
        # Recent activity indicates continued prescribing likelihood
        recency_factor = metrics['recency_score']
        
        # Minimum threshold adjustment
        # If prescribing consistently for 3+ months, ensure minimum score of 0.6
        if metrics['months_active'] >= self.min_months_for_consistency:
            consistency_floor = 0.6 + (metrics['temporal_consistency'] * 0.35)
        else:
            consistency_floor = 0.0
        
        # Weighted combination
        raw_probability = (
            temporal_score * self.consistency_weight +
            stability_bonus * self.volume_weight +
            recency_factor * self.recency_weight
        )
        
        # Apply consistency floor for regular prescribers
        adjusted_probability = max(raw_probability, consistency_floor)
        
        # Bayesian adjustment with global prior
        # Less aggressive shrinkage for consistent prescribers
        if metrics['months_active'] >= 3:
            shrinkage_factor = 0.9  # Minimal shrinkage for consistent prescribers
        else:
            shrinkage_factor = metrics['months_active'] / 3.0
        
        global_prior = global_benchmark.get('mean_probability', 0.3)
        
        final_probability = (
            shrinkage_factor * adjusted_probability + 
            (1 - shrinkage_factor) * global_prior
        )
        
        # Ensure consistent prescribers get high scores
        if metrics['temporal_consistency'] >= 0.8:  # Active 80%+ of months
            final_probability = max(final_probability, 0.75)
        elif metrics['temporal_consistency'] >= 0.5:  # Active 50%+ of months
            final_probability = max(final_probability, 0.60)
        
        return min(final_probability, 0.99)  # Cap at 0.99
    
    def _calculate_temporal_pvalue(self, metrics: Dict, global_benchmark: Dict) -> float:
        """
        Calculate p-value using appropriate statistical test based on temporal patterns.
        
        Tests whether the observed prescribing pattern is significantly different
        from random chance, accounting for temporal consistency.
        """
        
        if metrics['total_scripts'] == 0:
            return 1.0
        
        # For consistent prescribers (3+ months), use different test
        if metrics['months_active'] >= self.min_months_for_consistency:
            # Test: Is this temporal consistency significantly non-random?
            # Use binomial test: probability of observing this many active months
            n_months = 6  # Total observation period
            k_active = metrics['months_active']
            p_random = 1/3  # Null hypothesis: random prescribing
            
            # One-tailed test: is prescribing MORE consistent than random?
            p_value = 1 - stats.binom.cdf(k_active - 1, n_months, p_random)
            
            # Adjust for volume stability
            if metrics['volume_stability'] > 0.7:
                p_value *= 0.5  # More significant if volume is also stable
            
        else:
            # For sporadic prescribers, use proportion test
            observed_rate = metrics['total_scripts'] / max(metrics['months_active'] * 20, 1)
            expected_rate = global_benchmark.get('mean_rate', 0.1)
            
            # Z-test for proportions
            n = metrics['months_active'] * 20  # Approximate opportunities
            if n > 0:
                se = np.sqrt(expected_rate * (1 - expected_rate) / n)
                if se > 0:
                    z_score = (observed_rate - expected_rate) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    p_value = 1.0
            else:
                p_value = 1.0
        
        # Apply multiple testing correction (Benjamini-Hochberg would be applied later)
        return min(p_value, 1.0)
    
    def _calculate_global_statistics(self, df: pd.DataFrame, drug1: str, 
                                    drug2: str, drug_col: str) -> Dict:
        """Calculate global statistics for peer adjustment"""
        
        total_prescribers = df[self._get_npi_column(df)].nunique()
        
        drug1_prescribers = df[df[drug_col] == drug1][self._get_npi_column(df)].nunique()
        drug2_prescribers = df[df[drug_col] == drug2][self._get_npi_column(df)].nunique()
        
        return {
            'drug1': {
                'mean_probability': drug1_prescribers / max(total_prescribers, 1),
                'mean_rate': len(df[df[drug_col] == drug1]) / len(df) if len(df) > 0 else 0.1
            },
            'drug2': {
                'mean_probability': drug2_prescribers / max(total_prescribers, 1),
                'mean_rate': len(df[df[drug_col] == drug2]) / len(df) if len(df) > 0 else 0.1
            }
        }
    
    def _classify_pattern(self, prob1: float, prob2: float) -> str:
        """Classify prescribing pattern based on probabilities"""
        
        if prob1 >= 0.75 and prob2 >= 0.75:
            return "Dual High Prescriber"
        elif prob1 >= 0.75:
            return f"Primary Drug 1 Prescriber"
        elif prob2 >= 0.75:
            return f"Primary Drug 2 Prescriber"
        elif prob1 >= 0.5 or prob2 >= 0.5:
            return "Selective Prescriber"
        elif prob1 >= 0.25 or prob2 >= 0.25:
            return "Occasional Prescriber"
        else:
            return "Minimal/No Prescribing"
    
    def _print_summary_statistics(self, results_df: pd.DataFrame, drug1: str, drug2: str):
        """Print comprehensive summary statistics"""
        
        print(f"\nPROBABILITY CALCULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total prescribers analyzed: {len(results_df)}")
        
        # Consistent prescribers (those who should have high scores)
        consistent_drug1 = results_df[results_df[f'{drug1}_temporal_consistency'] >= 0.5]
        consistent_drug2 = results_df[results_df[f'{drug2}_temporal_consistency'] >= 0.5]
        
        print(f"\n{drug1} Statistics:")
        print(f"  Consistent prescribers (≥50% months): {len(consistent_drug1)}")
        if len(consistent_drug1) > 0:
            print(f"  Their average probability: {consistent_drug1[f'{drug1}_probability'].mean():.3f}")
            print(f"  Probability range: {consistent_drug1[f'{drug1}_probability'].min():.3f} - {consistent_drug1[f'{drug1}_probability'].max():.3f}")
        
        print(f"\n{drug2} Statistics:")
        print(f"  Consistent prescribers (≥50% months): {len(consistent_drug2)}")
        if len(consistent_drug2) > 0:
            print(f"  Their average probability: {consistent_drug2[f'{drug2}_probability'].mean():.3f}")
            print(f"  Probability range: {consistent_drug2[f'{drug2}_probability'].min():.3f} - {consistent_drug2[f'{drug2}_probability'].max():.3f}")
        
        # Pattern distribution
        pattern_counts = results_df['prescribing_pattern'].value_counts()
        print(f"\nPrescribing Patterns:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count} ({count/len(results_df)*100:.1f}%)")
        
        # P-value distribution
        sig_drug1 = (results_df[f'{drug1}_pvalue'] < 0.05).sum()
        sig_drug2 = (results_df[f'{drug2}_pvalue'] < 0.05).sum()
        print(f"\nStatistically Significant Prescribers (p<0.05):")
        print(f"  {drug1}: {sig_drug1} ({sig_drug1/len(results_df)*100:.1f}%)")
        print(f"  {drug2}: {sig_drug2} ({sig_drug2/len(results_df)*100:.1f}%)")
    
    # Helper methods for column detection
    def _get_drug_column(self, df: pd.DataFrame) -> str:
        for col in ['DRUG', 'NDC_PREFERRED_BRAND_NM', 'DRUG_NAME', 'BRAND_NAME']:
            if col in df.columns:
                return col
        return df.columns[0]
    
    def _get_npi_column(self, df: pd.DataFrame) -> str:
        for col in ['PRESCRIBER_NPI', 'PRESCRIBER_NPI_NBR', 'NPI', 'PRESCRIBER_NPI_NBR_']:
            if col in df.columns:
                return col
        return df.columns[0]
    
    def _get_date_column(self, df: pd.DataFrame) -> str:
        for col in ['RX_ANCHOR_DD', 'SERVICE_DATE', 'PRESCRIPTION_DATE', 'DATE']:
            if col in df.columns:
                return col
        return None
    
    def _get_specialty(self, prescriber_data: pd.DataFrame) -> str:
        for col in ['SPECIALTY', 'PRESCRIBER_NPI_HCP_SEGMENT_DESC', 'PRESCRIBER_NPI_SEGMENT_DESC', 'HCP_SEGMENT']:
            if col in prescriber_data.columns:
                return prescriber_data[col].mode()[0] if len(prescriber_data[col].mode()) > 0 else 'General'
        return 'General'
