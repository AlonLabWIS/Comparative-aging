import os
import pandas as pd
import glob

def process_csv_to_seed(input_csv_path, output_dir):
    # Read the CSV file
    df = pd.read_csv(input_csv_path, index_col=0)
    
    # Extract the mode_overall values for the required parameters
    seed_data = {
        'Eta': df.loc['eta', 'mode_overall'],
        'Beta': df.loc['beta', 'mode_overall'],
        'Epsilon': df.loc['epsilon', 'mode_overall'],
        'Xc': df.loc['xc', 'mode_overall'],
        'ExtH': df.loc['xc/eta', 'mode_overall']  # Using xc/eta as ExtH
    }
    
    # Create a DataFrame with a single row and 'Estimate' index
    seed_df = pd.DataFrame([seed_data], index=['Estimate'])
    
    # Create output filename
    base_name = os.path.basename(input_csv_path)
    output_name = f"seed_{base_name}"
    output_path = os.path.join(output_dir, output_name)
    
    # Save the seed file
    seed_df.to_csv(output_path)
    print(f"Created seed file: {output_path}")

def main():
    # Create output directory if it doesn't exist
    output_dir = "seeds_hetro01"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the results directory and its subdirectories
    results_dir = "final_baysian_01/results_CSVs"
    csv_files = glob.glob(os.path.join(results_dir, "**/*.csv"), recursive=True)
    
    # Process each CSV file
    for csv_file in csv_files:
        if not os.path.basename(csv_file).startswith('._'):  # Skip hidden files
            try:
                process_csv_to_seed(csv_file, output_dir)
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    main() 