import os
import glob
import configparser

def update_config_file(config_path):
    # Read the config file
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Update paths in DEFAULT section
    if 'DEFAULT' in config:
        # Update baysian01 to baysian02
        for key in config['DEFAULT']:
            if 'baysian01' in config['DEFAULT'][key]:
                config['DEFAULT'][key] = config['DEFAULT'][key].replace('baysian01', 'baysian02')
        
        # Update dataset folder name
        for key in config['DEFAULT']:
            if 'datasets' in config['DEFAULT'][key]:
                config['DEFAULT'][key] = config['DEFAULT'][key].replace('datasets', 'datasets_baysian_01')
        
        # Update run file
        config['DEFAULT']['run_file_mcmc'] = 'baysian02/run_mcmc_hetro.csh'
        
        # Add hetro field
        config['DEFAULT']['hetro'] = 'True'
        
        # Update seed file path
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        seed_name = f"seed_{config_name.replace('config_', '')}.csv"
        config['DEFAULT']['seed_file'] = f"hetro_seeds01/{seed_name}"
    
    # Update paths in SUBMISSION sections
    for section in config.sections():
        if section.startswith('SUBMISSION_'):
            if 'submission_folder' in config[section]:
                config[section]['submission_folder'] = config[section]['submission_folder'].replace('baysian01', 'baysian02')
    
    # Write the updated config
    with open(config_path, 'w') as f:
        config.write(f)
    
    print(f"Updated config file: {config_path}")

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