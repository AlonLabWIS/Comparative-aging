import os
import glob
import configparser
import shutil

def update_config_file(config_path):
    # Read the config file
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if 'DEFAULT' not in config:
        return
    
    # Get the base name without extension
    base_name = os.path.splitext(os.path.basename(config_path))[0]
    if not base_name.endswith('_hetro'):
        # Create new filename with _hetro suffix
        new_name = f"{base_name}_hetro.ini"
        
        # Update values in DEFAULT section
        if 'h5_file_name' in config['DEFAULT']:
            config['DEFAULT']['h5_file_name'] = f"{config['DEFAULT']['h5_file_name']}_hetro"
        
        if 'folder' in config['DEFAULT']:
            config['DEFAULT']['folder'] = config['DEFAULT']['folder'].replace('simulation_results/', 'simulation_results/') + '_hetro'
        
        if 'job_name' in config['DEFAULT']:
            config['DEFAULT']['job_name'] = f"{config['DEFAULT']['job_name']}_hetro"
        
        if 'name' in config['DEFAULT']:
            config['DEFAULT']['name'] = f"{config['DEFAULT']['name']}_hetro"
        
        if 'results_csv_file_name' in config['DEFAULT']:
            config['DEFAULT']['results_csv_file_name'] = f"{os.path.splitext(config['DEFAULT']['results_csv_file_name'])[0]}_hetro.csv"
        
        # Update seed file path
        if 'seed_file' in config['DEFAULT']:
            config['DEFAULT']['seed_file'] = f"baysian02/hetro_seeds01/seed_{base_name.replace('config_', '')}.csv"
        
        # Update submission folders
        for section in config.sections():
            if section.startswith('SUBMISSION_'):
                if 'submission_folder' in config[section]:
                    config[section]['submission_folder'] = config[section]['submission_folder'].replace('simulation_results/', 'simulation_results/') + '_hetro'
        
        # Create new directory in configs root
        new_dir = os.path.join("configs", f"{base_name}_hetro")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"Created new directory: {new_dir}")
        
        # Write the new config file in the new directory
        final_path = os.path.join(new_dir, new_name)
        with open(final_path, 'w') as f:
            config.write(f)
        print(f"Created new config file: {final_path}")

def main():
    # Get all config files in the configs directory and its subdirectories
    configs_dir = "configs"
    config_files = glob.glob(os.path.join(configs_dir, "**/*.ini"), recursive=True)
    
    # Process each config file
    for config_file in config_files:
        # Skip files in DROSOPHILA and SMURFS directories
        if 'DROSOPHILA' not in config_file and 'SMURFS' not in config_file:
            if not os.path.basename(config_file).startswith('._'):  # Skip hidden files
                try:
                    update_config_file(config_file)
                except Exception as e:
                    print(f"Error processing {config_file}: {str(e)}")

if __name__ == "__main__":
    main() 