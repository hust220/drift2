
#%% [markdown]
# # PDBbind Dataset Binding Affinity Analysis
# 
# This notebook analyzes binding affinity data from the PDBbind dataset and categorizes compounds into drug discovery stages based on their KD/Ki values.

#%% Imports and Setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, some features will be limited")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib only")

sys.path.append('../..')
from src.db_utils import db_connection

# Enable inline plotting for Jupyter (when available)
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except ImportError:
    pass  # Not in Jupyter environment

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

#%% Utility Functions
def convert_to_nanomolar(binding_value, binding_unit):
    """Convert binding value to nanomolar (nM) for consistent comparison"""
    if binding_value is None or binding_unit is None:
        return None
    
    unit_lower = binding_unit.lower().strip()
    
    # Convert to nanomolar (nM)
    if 'nm' in unit_lower:
        return binding_value
    elif 'um' in unit_lower or 'μm' in unit_lower:
        return binding_value * 1000.0  # uM to nM
    elif 'mm' in unit_lower:
        return binding_value * 1000000.0  # mM to nM
    elif unit_lower == 'm':
        return binding_value * 1000000000.0  # M to nM
    elif 'pm' in unit_lower:
        return binding_value / 1000.0  # pM to nM
    else:
        return None

def classify_compound_stage(binding_value_nm):
    """Classify compound into drug discovery stage based on KD/Ki value in nM"""
    if binding_value_nm is None:
        return "Unknown"
    
    if binding_value_nm >= 10000:  # >= 10 μM
        return "Hit (>10 μM)"
    elif 1 <= binding_value_nm < 10000:  # 1 nM - 10 μM
        return "Lead (1 nM - 10 μM)"
    elif 0.1 <= binding_value_nm < 1:  # 100 pM - 1 nM
        return "Candidate (100 pM - 1 nM)"
    else:  # < 100 pM
        return "Ultra-high affinity (<100 pM)"

#%% Data Loading and Processing
def analyze_binding_data():
    """Analyze binding affinity data from PDBbind dataset"""
    print("Analyzing PDBbind binding affinity data...")
    
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all binding data
        cursor.execute("""
            SELECT pdb_id, binding_type, binding_value, binding_unit, year, ligand_name, reference
            FROM pdbbind 
            WHERE binding_value IS NOT NULL 
            AND binding_unit IS NOT NULL
            ORDER BY binding_value
        """)
        
        data = cursor.fetchall()
        cursor.close()
    
    if not data:
        print("No binding data found in database")
        return
    
    print(f"Found {len(data)} entries with binding data")
    
    # Process data
    results = []
    conversion_stats = {"converted": 0, "failed": 0, "units": Counter()}
    
    for pdb_id, binding_type, binding_value, binding_unit, year, ligand_name, reference in data:
        conversion_stats["units"][binding_unit] += 1
        
        # Convert to nM
        binding_value_nm = convert_to_nanomolar(binding_value, binding_unit)
        
        if binding_value_nm is not None:
            conversion_stats["converted"] += 1
            stage = classify_compound_stage(binding_value_nm)
            
            results.append({
                'pdb_id': pdb_id,
                'binding_type': binding_type,
                'original_value': binding_value,
                'original_unit': binding_unit,
                'value_nm': binding_value_nm,
                'stage': stage,
                'year': year,
                'ligand_name': ligand_name
            })
        else:
            conversion_stats["failed"] += 1
    
    print(f"\nUnit conversion statistics:")
    print(f"Successfully converted: {conversion_stats['converted']}")
    print(f"Failed to convert: {conversion_stats['failed']}")
    print(f"Unit distribution:")
    for unit, count in conversion_stats["units"].most_common():
        print(f"  {unit}: {count}")
    
    if not results:
        print("No data could be converted to nM")
        return
    
    # Analyze data
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        return analyze_with_pandas(df)
    else:
        return analyze_without_pandas(results)

#%% Statistical Analysis Functions
def analyze_with_pandas(df):
    """Analyze data using pandas"""
    # Stage distribution analysis
    stage_counts = df['stage'].value_counts()
    print(f"\nDrug Discovery Stage Distribution:")
    print("=" * 50)
    total = len(df)
    for stage, count in stage_counts.items():
        percentage = (count / total) * 100
        print(f"{stage:<30}: {count:>6} ({percentage:5.1f}%)")
    
    # Binding type analysis
    binding_type_counts = df['binding_type'].value_counts()
    print(f"\nBinding Type Distribution:")
    print("=" * 30)
    for binding_type, count in binding_type_counts.items():
        percentage = (count / total) * 100
        print(f"{binding_type:<10}: {count:>6} ({percentage:5.1f}%)")
    
    # Statistical summary
    print(f"\nBinding Affinity Statistics (nM):")
    print("=" * 40)
    print(f"Count: {len(df)}")
    print(f"Mean: {df['value_nm'].mean():.2f} nM")
    print(f"Median: {df['value_nm'].median():.2f} nM")
    print(f"Min: {df['value_nm'].min():.2f} nM")
    print(f"Max: {df['value_nm'].max():.2f} nM")
    print(f"Std: {df['value_nm'].std():.2f} nM")
    
    # Percentiles for different stages
    print(f"\nPercentile Analysis:")
    print("=" * 30)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(df['value_nm'], p)
        stage = classify_compound_stage(value)
        print(f"{p:2d}th percentile: {value:8.2f} nM ({stage})")
    
    return df

def analyze_without_pandas(results):
    """Analyze data without pandas using basic Python"""
    # Stage distribution analysis
    stage_counts = Counter(item['stage'] for item in results)
    print(f"\nDrug Discovery Stage Distribution:")
    print("=" * 50)
    total = len(results)
    for stage, count in stage_counts.most_common():
        percentage = (count / total) * 100
        print(f"{stage:<30}: {count:>6} ({percentage:5.1f}%)")
    
    # Binding type analysis
    binding_type_counts = Counter(item['binding_type'] for item in results)
    print(f"\nBinding Type Distribution:")
    print("=" * 30)
    for binding_type, count in binding_type_counts.most_common():
        percentage = (count / total) * 100
        print(f"{binding_type:<10}: {count:>6} ({percentage:5.1f}%)")
    
    # Statistical summary
    values_nm = [item['value_nm'] for item in results]
    print(f"\nBinding Affinity Statistics (nM):")
    print("=" * 40)
    print(f"Count: {len(values_nm)}")
    print(f"Mean: {np.mean(values_nm):.2f} nM")
    print(f"Median: {np.median(values_nm):.2f} nM")
    print(f"Min: {np.min(values_nm):.2f} nM")
    print(f"Max: {np.max(values_nm):.2f} nM")
    print(f"Std: {np.std(values_nm):.2f} nM")
    
    # Percentiles for different stages
    print(f"\nPercentile Analysis:")
    print("=" * 30)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(values_nm, p)
        stage = classify_compound_stage(value)
        print(f"{p:2d}th percentile: {value:8.2f} nM ({stage})")
    
    return results

#%% Visualization Functions
def create_visualizations(data):
    """Create visualizations for the binding data analysis"""
    if data is None or (HAS_PANDAS and data.empty) or (not HAS_PANDAS and len(data) == 0):
        print("No data available for visualization")
        return
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PDBbind Dataset Binding Affinity Analysis', fontsize=16)
        
        if HAS_PANDAS:
            # Use pandas data
            df = data
            stage_counts = df['stage'].value_counts()
            binding_type_counts = df['binding_type'].value_counts()
            values_nm = df['value_nm'].values
        else:
            # Use list of dicts
            stage_counts = Counter(item['stage'] for item in data)
            binding_type_counts = Counter(item['binding_type'] for item in data)
            values_nm = np.array([item['value_nm'] for item in data])
        
        # 1. Stage distribution pie chart
        if HAS_PANDAS:
            axes[0, 0].pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%', startangle=90)
        else:
            labels, counts = zip(*stage_counts.most_common())
            axes[0, 0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Drug Discovery Stage Distribution')
        
        # 2. Binding affinity histogram (log scale)
        axes[0, 1].hist(np.log10(values_nm), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('log10(Binding Affinity, nM)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Binding Affinity Distribution (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add vertical lines for stage boundaries
        stage_boundaries = [np.log10(0.1), np.log10(1), np.log10(10000)]
        stage_labels = ['100 pM', '1 nM', '10 μM']
        for boundary, label in zip(stage_boundaries, stage_labels):
            axes[0, 1].axvline(boundary, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].text(boundary, axes[0, 1].get_ylim()[1] * 0.9, label, 
                           rotation=90, verticalalignment='top')
        
        # 3. Binding type distribution
        if HAS_PANDAS:
            axes[1, 0].bar(binding_type_counts.index, binding_type_counts.values)
        else:
            labels, counts = zip(*binding_type_counts.most_common())
            axes[1, 0].bar(labels, counts)
        axes[1, 0].set_xlabel('Binding Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Binding Type Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Simple scatter plot instead of heatmap if no seaborn
        if HAS_PANDAS and HAS_SEABORN:
            stage_binding_crosstab = pd.crosstab(df['stage'], df['binding_type'])
            sns.heatmap(stage_binding_crosstab, annot=True, fmt='d', ax=axes[1, 1], cmap='YlOrRd')
            axes[1, 1].set_title('Stage vs Binding Type Distribution')
        else:
            # Simple text-based visualization
            axes[1, 1].text(0.1, 0.5, 'Stage vs Binding Type\n(Detailed heatmap requires\npandas and seaborn)', 
                           transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = '/home/tyq4zn/drift2/data/pdbbind/binding_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Show plot in Jupyter environment
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Continuing without visualizations...")

#%% Detailed Analysis Functions
def detailed_stage_analysis(data):
    """Provide detailed analysis for each drug discovery stage"""
    if data is None or (HAS_PANDAS and data.empty) or (not HAS_PANDAS and len(data) == 0):
        return
    
    print("\n" + "="*60)
    print("DETAILED STAGE ANALYSIS")
    print("="*60)
    
    stages = ["Ultra-high affinity (<100 pM)", "Candidate (100 pM - 1 nM)", 
              "Lead (1 nM - 10 μM)", "Hit (>10 μM)"]
    
    for stage in stages:
        if HAS_PANDAS:
            stage_data = data[data['stage'] == stage]
            if len(stage_data) == 0:
                continue
            values_nm = stage_data['value_nm'].values
            binding_types = stage_data['binding_type'].value_counts()
            examples = stage_data.nsmallest(5, 'value_nm')[['pdb_id', 'value_nm', 'binding_type', 'ligand_name']]
        else:
            stage_data = [item for item in data if item['stage'] == stage]
            if len(stage_data) == 0:
                continue
            values_nm = [item['value_nm'] for item in stage_data]
            binding_types = Counter(item['binding_type'] for item in stage_data)
            # Get top 5 by affinity
            examples = sorted(stage_data, key=lambda x: x['value_nm'])[:5]
            
        print(f"\n{stage}")
        print("-" * len(stage))
        print(f"Count: {len(stage_data)}")
        total_count = len(data) if not HAS_PANDAS else len(data)
        print(f"Percentage: {len(stage_data)/total_count*100:.1f}%")
        print(f"Affinity range: {np.min(values_nm):.2f} - {np.max(values_nm):.2f} nM")
        print(f"Mean affinity: {np.mean(values_nm):.2f} nM")
        print(f"Median affinity: {np.median(values_nm):.2f} nM")
        
        # Binding type distribution within stage
        print(f"Binding types:")
        if HAS_PANDAS:
            for bt, count in binding_types.items():
                print(f"  {bt}: {count} ({count/len(stage_data)*100:.1f}%)")
        else:
            for bt, count in binding_types.most_common():
                print(f"  {bt}: {count} ({count/len(stage_data)*100:.1f}%)")
        
        # Show some examples
        print(f"Examples (top 5 by affinity):")
        if HAS_PANDAS:
            for _, row in examples.iterrows():
                ligand = row['ligand_name'] if pd.notna(row['ligand_name']) else 'Unknown'
                print(f"  {row['pdb_id']}: {row['value_nm']:.2f} nM ({row['binding_type']}) - {ligand}")
        else:
            for item in examples:
                ligand = item['ligand_name'] if item['ligand_name'] else 'Unknown'
                print(f"  {item['pdb_id']}: {item['value_nm']:.2f} nM ({item['binding_type']}) - {ligand}")

#%% Export Functions
def export_analysis_results(data):
    """Export analysis results to CSV files"""
    if data is None or (HAS_PANDAS and data.empty) or (not HAS_PANDAS and len(data) == 0):
        return
    
    if not HAS_PANDAS:
        print("CSV export requires pandas. Results available in console output only.")
        return
    
    output_dir = '/home/tyq4zn/drift2/data/pdbbind'
    
    # Export full dataset with classifications
    full_output_path = os.path.join(output_dir, 'pdbbind_binding_analysis.csv')
    data.to_csv(full_output_path, index=False)
    print(f"Full analysis exported to: {full_output_path}")
    
    # Export summary statistics
    summary_data = []
    for stage in data['stage'].unique():
        stage_data = data[data['stage'] == stage]
        summary_data.append({
            'stage': stage,
            'count': len(stage_data),
            'percentage': len(stage_data)/len(data)*100,
            'min_nm': stage_data['value_nm'].min(),
            'max_nm': stage_data['value_nm'].max(),
            'mean_nm': stage_data['value_nm'].mean(),
            'median_nm': stage_data['value_nm'].median(),
            'std_nm': stage_data['value_nm'].std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = os.path.join(output_dir, 'pdbbind_stage_summary.csv')
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Stage summary exported to: {summary_output_path}")

#%% Main Analysis - Run All Steps
def main():
    """Main analysis function"""
    print("PDBbind Dataset Binding Affinity Analysis")
    print("=" * 50)
    
    # Perform main analysis
    data = analyze_binding_data()
    
    if data is not None:
        data_len = len(data) if not HAS_PANDAS else len(data)
        if data_len > 0:
            # Detailed stage analysis
            detailed_stage_analysis(data)
            
            # Create visualizations
            create_visualizations(data)
            
            # Export results
            export_analysis_results(data)
            
            print(f"\nAnalysis complete! Processed {data_len} compounds.")
            files_generated = ["- binding_analysis.png (visualization)"]
            if HAS_PANDAS:
                files_generated.extend([
                    "- pdbbind_binding_analysis.csv (full data)",
                    "- pdbbind_stage_summary.csv (summary statistics)"
                ])
            print("Files generated:")
            for file_info in files_generated:
                print(file_info)
        else:
            print("No data available for analysis")
    else:
        print("No data available for analysis")

#%% Interactive Analysis - Load Data
# Run this cell to load and analyze the data
data = analyze_binding_data()

#%% Interactive Analysis - Create Visualizations  
# Run this cell to create and display visualizations
if data is not None:
    create_visualizations(data)

#%% Interactive Analysis - Detailed Stage Analysis
# Run this cell for detailed analysis of each drug discovery stage
if data is not None:
    detailed_stage_analysis(data)

#%% Interactive Analysis - Export Results
# Run this cell to export results to CSV files
if data is not None:
    export_analysis_results(data)

#%% Script Execution
# This will run when the file is executed as a script
if __name__ == "__main__":
    main()
