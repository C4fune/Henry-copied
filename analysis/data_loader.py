import pandas as pd
from google.cloud import bigquery
from typing import Optional, List, Tuple, Dict, Any
from config import *


class DataLoader:
    def __init__(self):
        self.client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    
    def load_pharmacy_data(self, 
                          drug_filter: Optional[List[str]] = None,
                          state_filter: Optional[List[str]] = None,
                          time_range: Optional[Tuple[str, str]] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """Load prescription claims data"""
        
        query = f"""
        SELECT 
            PRESCRIBER_NPI_NBR,
            PRESCRIBER_NPI_NM,
            PRESCRIBER_NPI_HCP_SEGMENT_DESC,
            PRESCRIBER_NPI_STATE_CD,
            PRESCRIBER_NPI_ZIP5_CD,
            RX_ANCHOR_DD,
            SERVICE_DATE_DD,
            NDC,
            NDC_PREFERRED_BRAND_NM,
            NDC_DRUG_CLASS_NM,
            NDC_DRUG_SUBCLASS_NM,
            PHARMACY_NPI_STATE_CD,
            PAYER_PLAN_CHANNEL_NM,
            DISPENSED_QUANTITY_VAL,
            DAYS_SUPPLY_VAL,
            TOTAL_PAID_AMT,
            PATIENT_TO_PAY_AMT,
            EXTRACT(YEAR FROM RX_ANCHOR_DD) as rx_year,
            EXTRACT(MONTH FROM RX_ANCHOR_DD) as rx_month
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{PHARMACY_TABLE}`
        WHERE 1=1
        """
        
        if drug_filter:
            drug_list = "', '".join(drug_filter)
            query += f" AND UPPER(NDC_PREFERRED_BRAND_NM) IN ('{drug_list.upper()}')"
        
        if state_filter:
            state_list = "', '".join(state_filter)
            query += f" AND PRESCRIBER_NPI_STATE_CD IN ('{state_list}')"
        
        if time_range:
            query += f" AND RX_ANCHOR_DD BETWEEN '{time_range[0]}' AND '{time_range[1]}'"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Warning: BigQuery access failed")
            return pd.DataFrame()
    
    def load_medical_claims(self,
                           diagnosis_filter: Optional[List[str]] = None,
                           procedure_filter: Optional[List[str]] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Load medical claims data"""
        
        query = f"""
        SELECT 
            PRESCRIBER_NPI_NBR,
            MEDICAL_SERVICE_DD,
            PRIMARY_DX_CD,
            PRIMARY_DX_DESC,
            SECONDARY_DX_CD,
            SECONDARY_DX_DESC,
            PROCEDURE_CD,
            PROCEDURE_DESC,
            PLACE_OF_SERVICE_DESC,
            PAYER_PLAN_CHANNEL_NM,
            TOTAL_PAID_AMT,
            PATIENT_TO_PAY_AMT,
            EXTRACT(YEAR FROM MEDICAL_SERVICE_DD) as service_year,
            EXTRACT(MONTH FROM MEDICAL_SERVICE_DD) as service_month
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{MEDICAL_TABLE}`
        WHERE 1=1
        """
        
        if diagnosis_filter:
            dx_list = "', '".join(diagnosis_filter)
            query += f" AND (PRIMARY_DX_CD IN ('{dx_list}') OR SECONDARY_DX_CD IN ('{dx_list}'))"
        
        if procedure_filter:
            proc_list = "', '".join(procedure_filter)
            query += f" AND PROCEDURE_CD IN ('{proc_list}')"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Warning: BigQuery access failed")
            return pd.DataFrame()
    
    def load_hcp_data(self, 
                     specialty_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load healthcare provider biographical data"""
        
        query = f"""
        SELECT 
            npi,
            provider_name,
            provider_first_name,
            provider_last_name,
            provider_middle_name,
            provider_credential,
            provider_gender,
            provider_entity_type,
            provider_street_address_1,
            provider_city,
            provider_state,
            provider_zip_code,
            provider_country,
            provider_taxonomy_code,
            provider_taxonomy_description,
            is_sole_proprietor,
            is_organization_subpart,
            parent_organization_name,
            organizational_name,
            provider_enumeration_date,
            last_update_date,
            deactivation_date,
            reactivation_date
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.providers_bio`
        WHERE 1=1
        """
        
        if specialty_filter:
            spec_list = "', '".join(specialty_filter)
            query += f" AND provider_taxonomy_description IN ('{spec_list}')"
        
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Warning: BigQuery access failed")
            return pd.DataFrame()
    
    def load_provider_payments(self,
                              time_range: Optional[Tuple[str, str]] = None,
                              payment_type_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load provider payment data"""
        
        query = f"""
        SELECT 
            physician_npi,
            physician_first_name,
            physician_last_name,
            physician_specialty,
            physician_primary_type,
            recipient_city,
            recipient_state,
            payer_name,
            payer_id,
            product_name,
            product_category,
            payment_amount,
            payment_date,
            payment_type,
            nature_of_payment,
            dispute_status,
            product_indicator,
            associated_drug_or_device,
            EXTRACT(YEAR FROM payment_date) as payment_year,
            EXTRACT(MONTH FROM payment_date) as payment_month
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.provider_payments`
        WHERE 1=1
        """
        
        if time_range:
            query += f" AND payment_date BETWEEN '{time_range[0]}' AND '{time_range[1]}'"
        
        if payment_type_filter:
            type_list = "', '".join(payment_type_filter)
            query += f" AND payment_type IN ('{type_list}')"
        
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Warning: BigQuery access failed")
            return pd.DataFrame()
    
    def load_npi_doctors(self,
                        state_filter: Optional[List[str]] = None,
                        specialty_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load NPI doctor registry data"""
        
        query = f"""
        SELECT 
            npi,
            entity_type_code,
            provider_organization_name,
            provider_last_name,
            provider_first_name,
            provider_middle_name,
            provider_name_suffix,
            provider_credential,
            provider_gender_code,
            provider_business_practice_location_address_city_name,
            provider_business_practice_location_address_state_name,
            provider_business_practice_location_address_postal_code,
            healthcare_provider_taxonomy_code_1,
            healthcare_provider_primary_taxonomy_switch_1,
            healthcare_provider_taxonomy_group_1,
            is_sole_proprietor,
            is_organization_subpart,
            parent_organization_lbn,
            authorized_official_first_name,
            authorized_official_last_name,
            authorized_official_title_or_position,
            authorized_official_telephone_number
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.us_npi_doctors`
        WHERE entity_type_code = 1
        """
        
        if state_filter:
            state_list = "', '".join(state_filter)
            query += f" AND provider_business_practice_location_address_state_name IN ('{state_list}')"
        
        if specialty_filter:
            spec_list = "', '".join(specialty_filter)
            query += f" AND healthcare_provider_taxonomy_group_1 IN ('{spec_list}')"
        
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Warning: BigQuery access failed")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        
        summary_query = f"""
        WITH rx_summary AS (
            SELECT COUNT(*) as rx_count,
                   COUNT(DISTINCT PRESCRIBER_NPI_NBR) as prescriber_count,
                   COUNT(DISTINCT NDC_PREFERRED_BRAND_NM) as drug_count
            FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{PHARMACY_TABLE}`
        ),
        medical_summary AS (
            SELECT COUNT(*) as medical_count,
                   COUNT(DISTINCT PRIMARY_DX_CD) as diagnosis_count
            FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{MEDICAL_TABLE}`
        )
        SELECT * FROM rx_summary, medical_summary
        """
        
        try:
            result = self.client.query(summary_query).to_dataframe()
            return result.iloc[0].to_dict() if len(result) > 0 else {}
        except:
            return {}
    
    def get_available_datasets(self) -> List[str]:
        """Return list of available datasets"""
        return ['rx_claims', 'medical_claims', 'providers_bio', 'provider_payments', 'us_npi_doctors']