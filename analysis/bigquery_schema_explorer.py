"""
Explore and document all BigQuery tables and their schemas
"""
import pandas as pd
from google.cloud import bigquery
from config import BIGQUERY_PROJECT_ID, BIGQUERY_DATASET
import json

class BigQuerySchemaExplorer:
    def __init__(self):
        self.client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        self.project_id = BIGQUERY_PROJECT_ID
        self.dataset = BIGQUERY_DATASET
    
    def explore_all_tables(self):
        """Get comprehensive information about all tables in the dataset"""
        
        # List all tables in the dataset
        tables_query = f"""
        SELECT table_name, row_count, size_bytes
        FROM `{self.project_id}.{self.dataset}.INFORMATION_SCHEMA.TABLES`
        """
        
        try:
            tables = self.client.query(tables_query).to_dataframe()
            print(f"\n{'='*80}")
            print(f"BIGQUERY DATASET ANALYSIS: {self.project_id}.{self.dataset}")
            print(f"{'='*80}")
            print(f"\nTables found: {len(tables)}")
            
            for _, table_info in tables.iterrows():
                table_name = table_info['table_name']
                row_count = table_info['row_count']
                size_mb = table_info['size_bytes'] / (1024*1024)
                
                print(f"\n{'='*60}")
                print(f"TABLE: {table_name}")
                print(f"Rows: {row_count:,}")
                print(f"Size: {size_mb:.2f} MB")
                print(f"{'='*60}")
                
                # Get column information
                columns_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM `{self.project_id}.{self.dataset}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
                """
                
                columns = self.client.query(columns_query).to_dataframe()
                print("\nColumns:")
                for _, col in columns.iterrows():
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    print(f"  - {col['column_name']}: {col['data_type']} ({nullable})")
                
                # Get sample data and statistics
                self._analyze_table_content(table_name)
        
        except Exception as e:
            print(f"Error exploring tables: {e}")
            self._explore_alternative_method()
    
    def _analyze_table_content(self, table_name):
        """Analyze specific content patterns in a table"""
        
        try:
            # For prescription data, analyze prescribing patterns
            if 'rx' in table_name.lower() or 'claim' in table_name.lower():
                analysis_query = f"""
                WITH prescriber_stats AS (
                    SELECT 
                        PRESCRIBER_NPI_NBR,
                        COUNT(DISTINCT NDC_PREFERRED_BRAND_NM) as drug_diversity,
                        COUNT(*) as total_scripts,
                        COUNT(DISTINCT DATE(RX_ANCHOR_DD)) as active_days,
                        MIN(RX_ANCHOR_DD) as first_script,
                        MAX(RX_ANCHOR_DD) as last_script
                    FROM `{self.project_id}.{self.dataset}.{table_name}`
                    GROUP BY PRESCRIBER_NPI_NBR
                    LIMIT 1000
                )
                SELECT 
                    AVG(drug_diversity) as avg_drug_diversity,
                    AVG(total_scripts) as avg_scripts_per_doc,
                    AVG(active_days) as avg_active_days,
                    MIN(first_script) as earliest_date,
                    MAX(last_script) as latest_date
                FROM prescriber_stats
                """
                
                stats = self.client.query(analysis_query).to_dataframe()
                if not stats.empty:
                    print("\nPrescribing Pattern Statistics:")
                    print(f"  Average drugs per prescriber: {stats['avg_drug_diversity'].iloc[0]:.1f}")
                    print(f"  Average scripts per prescriber: {stats['avg_scripts_per_doc'].iloc[0]:.1f}")
                    print(f"  Average active days: {stats['avg_active_days'].iloc[0]:.1f}")
                    print(f"  Date range: {stats['earliest_date'].iloc[0]} to {stats['latest_date'].iloc[0]}")
        
        except Exception as e:
            print(f"  Could not analyze content: {e}")
    
    def _explore_alternative_method(self):
        """Alternative method using direct table queries"""
        
        print("\nAttempting alternative exploration method...")
        
        # Known tables based on documentation
        known_tables = [
            'rx_claims',
            'medical_claims', 
            'providers_bio',
            'provider_payments',
            'us_npi_doctors'
        ]
        
        for table in known_tables:
            try:
                # Get schema
                query = f"""
                SELECT *
                FROM `{self.project_id}.{self.dataset}.{table}`
                LIMIT 0
                """
                
                df = self.client.query(query).to_dataframe()
                print(f"\n{table} columns:")
                for col in df.columns:
                    print(f"  - {col}: {df[col].dtype}")
                
                # Get row count
                count_query = f"""
                SELECT COUNT(*) as count
                FROM `{self.project_id}.{self.dataset}.{table}`
                """
                
                count_df = self.client.query(count_query).to_dataframe()
                print(f"  Row count: {count_df['count'].iloc[0]:,}")
                
            except Exception as e:
                print(f"\n{table}: Could not access - {e}")
    
    def analyze_temporal_patterns(self):
        """Analyze temporal consistency in prescribing data"""
        
        query = f"""
        WITH monthly_prescribing AS (
            SELECT 
                PRESCRIBER_NPI_NBR,
                NDC_PREFERRED_BRAND_NM,
                DATE_TRUNC(RX_ANCHOR_DD, MONTH) as month,
                COUNT(*) as monthly_scripts
            FROM `{self.project_id}.{self.dataset}.rx_claims`
            WHERE RX_ANCHOR_DD >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
            GROUP BY 1, 2, 3
        ),
        consistency_scores AS (
            SELECT 
                PRESCRIBER_NPI_NBR,
                NDC_PREFERRED_BRAND_NM,
                COUNT(DISTINCT month) as months_active,
                AVG(monthly_scripts) as avg_monthly_scripts,
                STDDEV(monthly_scripts) as script_variability,
                -- Consistency score: high if prescribed in many months with low variability
                COUNT(DISTINCT month) / 6.0 as temporal_consistency,
                CASE 
                    WHEN STDDEV(monthly_scripts) = 0 THEN 1.0
                    ELSE 1 / (1 + STDDEV(monthly_scripts) / AVG(monthly_scripts))
                END as volume_consistency
            FROM monthly_prescribing
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT month) >= 3  -- At least 3 months of data
        )
        SELECT 
            AVG(temporal_consistency) as avg_temporal_consistency,
            AVG(volume_consistency) as avg_volume_consistency,
            COUNT(DISTINCT PRESCRIBER_NPI_NBR) as consistent_prescribers,
            COUNT(DISTINCT NDC_PREFERRED_BRAND_NM) as drugs_analyzed
        FROM consistency_scores
        WHERE temporal_consistency >= 0.5  -- Active at least 50% of months
        """
        
        try:
            results = self.client.query(query).to_dataframe()
            print("\n" + "="*60)
            print("TEMPORAL CONSISTENCY ANALYSIS")
            print("="*60)
            print(f"Average temporal consistency: {results['avg_temporal_consistency'].iloc[0]:.3f}")
            print(f"Average volume consistency: {results['avg_volume_consistency'].iloc[0]:.3f}")
            print(f"Consistent prescribers: {results['consistent_prescribers'].iloc[0]:,}")
            print(f"Drugs analyzed: {results['drugs_analyzed'].iloc[0]:,}")
        except Exception as e:
            print(f"Could not analyze temporal patterns: {e}")

if __name__ == "__main__":
    explorer = BigQuerySchemaExplorer()
    explorer.explore_all_tables()
    explorer.analyze_temporal_patterns()
