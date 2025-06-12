import os
import papermill as pm
import subprocess
import json
import datetime

# --- CONFIGURATION ---
NOTEBOOK_PATH = "mcmc_analysis_systemic_01/mcmc_analysis_sys.ipynb"
OUTPUT_DIR = "notebook_runs"
PARAMS_FILE = "mcmc_analysis_systemic_01/params.json"

# Get absolute paths
BASE_DIR = os.path.abspath(os.getcwd())
POSTERIOR_DIR = os.path.join(BASE_DIR, "posterior_csvs_baysian01")
PRODUCTS_DIR = os.path.join(BASE_DIR, "final_baysian_01")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets_baysian_01")
SEED_DIR = os.path.join(BASE_DIR, "seeds_baysian_01")


# --- LOAD PARAMETER SETS ---
with open(PARAMS_FILE, "r") as f:
    param_sets = json.load(f)

# Create all output directories and subfolders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(POSTERIOR_DIR, exist_ok=True)
os.makedirs(PRODUCTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PRODUCTS_DIR, 'results_csvs'), exist_ok=True)
os.makedirs(os.path.join(PRODUCTS_DIR, 'html_3d_plots'), exist_ok=True)


# Print current working directory
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Posterior directory: {POSTERIOR_DIR}")
print(f"Products directory: {PRODUCTS_DIR}")
print(f"Datasets directory: {DATASETS_DIR}")
print(f"Seed directory: {SEED_DIR}")

for i, params in enumerate(param_sets):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{params['file']}_run_{i+1}_{timestamp}"
    subfolder = params.get('subfolder', None)  # Explicitly handle None case

    if subfolder:
        executed_nb = os.path.join(OUTPUT_DIR, subfolder, f"{run_name}.ipynb")
        pdf_out = os.path.join(OUTPUT_DIR, subfolder, f"{run_name}.pdf")
    else:
        executed_nb = os.path.join(OUTPUT_DIR, f"{run_name}.ipynb")
        pdf_out = os.path.join(OUTPUT_DIR, f"{run_name}.pdf")

    # Add the absolute paths to the parameters
    params['folder'] = POSTERIOR_DIR
    params['Products_folder'] = PRODUCTS_DIR
    params['datasets_folder'] = DATASETS_DIR
    params['seed_folder'] = SEED_DIR

    print(f"\nProcessing run {i+1} with parameters:")
    print(f"File: {params['file']}")
    print(f"Scale: {params['scale']}")
    print(f"Time Unit: {params['TIME_UNIT']}")
    print(f"Number of bins: {params['nbins']}")
    print(f"Folder: {params['folder']}")
    print(f"Products folder: {params['Products_folder']}")
    print(f"Datasets folder: {params['datasets_folder']}")
    print(f"Seed folder: {params['seed_folder']}")
    print(f"Subfolder: {params.get('subfolder')}")

    # Create subfolders if they don't exist
    if subfolder:
        os.makedirs(os.path.join(POSTERIOR_DIR, subfolder), exist_ok=True)
        os.makedirs(os.path.join(PRODUCTS_DIR, subfolder), exist_ok=True)
        os.makedirs(os.path.join(DATASETS_DIR, subfolder), exist_ok=True)
        os.makedirs(os.path.join(SEED_DIR, subfolder), exist_ok=True)
        os.makedirs(os.path.join(PRODUCTS_DIR, 'results_csvs', subfolder), exist_ok=True)
        os.makedirs(os.path.join(PRODUCTS_DIR, 'html_3d_plots', subfolder), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)

    # Check if the file exists
    expected_path = os.path.join(POSTERIOR_DIR, params['file'])
    print(f"Looking for file at: {expected_path}")
    print(f"File exists: {os.path.exists(expected_path)}")

    # 1. Run notebook with parameters
    try:
        pm.execute_notebook(
            NOTEBOOK_PATH,
            executed_nb,
            parameters=params
        )
        print(f"Notebook execution completed: {executed_nb}")

        # 2. Export to PDF (code cells hidden)
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "pdf",
            "--TemplateExporter.exclude_input=True",
            "--TemplateExporter.exclude_output_prompt=True",
            "--TemplateExporter.exclude_input_prompt=True",
            executed_nb
        ], check=True)

        # 3. Move PDF to desired location
        generated_pdf = executed_nb.replace(".ipynb", ".pdf")
        if os.path.exists(generated_pdf):
            os.rename(generated_pdf, pdf_out)
            print(f"PDF created successfully: {pdf_out}")
        else:
            print(f"Warning: PDF was not generated for {executed_nb}")

    except Exception as e:
        print(f"Error processing run {i+1}: {str(e)}")
        continue

print("\nAll runs completed!")