import json
import sys
sys.path.append('.')
import pandas as pd

from dynamic_query_processor import DynamicQueryProcessor

def print_header():
    print("="*70)
    print("PHARMACEUTICAL ANALYTICS SYSTEM")
    print("="*70)

def format_results(results: dict):
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    if results.get("error"):
        print(f"\nError: {results['error']}")
        return
    
    if results.get("insights"):
        print(f"\nINSIGHTS:")
        print("-"*60)
        print(results['insights'])
    
    if results.get("technical_data"):
        print(f"\n\nTECHNICAL DATA (Evidence/Proof):")
        print("-"*60)
        
        tech_data = results['technical_data']
        
        if 'data_quality' in tech_data:
            print(f"\nData Quality:")
            for key, value in tech_data['data_quality'].items():
                print(f"  • {key}: {value}")
        
        if 'statistical_evidence' in tech_data:
            print(f"\nStatistical Evidence:")
            for test_name, test_data in tech_data['statistical_evidence'].items():
                if isinstance(test_data, dict):
                    print(f"  • {test_name}:")
                    print(f"    - p-value: {test_data.get('value', 'N/A')}")
                    print(f"    - Significant: {test_data.get('significant', 'N/A')}")
        
        if 'key_findings' in tech_data:
            print(f"\nKey Findings:")
            for metric_name, metric_value in tech_data['key_findings'].items():
                if isinstance(metric_value, (int, float)):
                    print(f"  • {metric_name}: {metric_value:,.2f}")
                elif isinstance(metric_value, list) and len(metric_value) < 5:
                    print(f"  • {metric_name}: {metric_value}")
        
        if 'methodology' in tech_data:
            print(f"\nMethodology:")
            if isinstance(tech_data['methodology'], dict):
                for key, value in tech_data['methodology'].items():
                    print(f"  • {key}: {value}")
    
    if results.get("results"):
        print(f"\n\nRESULTS SUMMARY:")
        print("-"*60)
        
        for key, value in results['results'].items():
            if isinstance(value, pd.DataFrame):
                print(f"{key}: DataFrame with shape {value.shape}")
            elif isinstance(value, dict) and len(str(value)) < 200:
                print(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {type(value).__name__}")
    
    if results.get("visualizations"):
        print(f"\n\nVISUALIZATIONS:")
        print("-"*60)
        for viz in results['visualizations']:
            print(f"  ✓ {viz}")
    
    if results.get("text_report"):
        print(f"\n\nTEXT REPORT:")
        print("-"*60)
        print(f"  ✓ {results['text_report']}")
    
    if results.get("output_folder"):
        print(f"\n\n📁 ALL OUTPUTS SAVED TO:")
        print("-"*60)
        print(f"  {results['output_folder']}")
    
    print("="*70)

def main():
    import sys
    
    print_header()
    processor = DynamicQueryProcessor()
    
    # Check if query is provided as command line argument
    if len(sys.argv) > 1:
        # Single query mode
        query = ' '.join(sys.argv[1:])
        print(f"\nProcessing query: {query}\n")
        results = processor.process(query)
        format_results(results)
        return
    
    # Check if input is being piped
    if not sys.stdin.isatty():
        try:
            query = sys.stdin.read().strip()
            if query:
                print(f"\nProcessing query: {query}\n")
                results = processor.process(query)
                format_results(results)
                return
        except:
            pass
    
    # Interactive mode
    print("\nReady for queries. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting...")
                break
            
            if not query:
                continue
            
            results = processor.process(query)
            format_results(results)
            
        except EOFError:
            # Handle EOF gracefully
            print("\n\nExiting...")
            break
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()