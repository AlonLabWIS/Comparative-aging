import os
import glob

def remove_thumbnail_files():
    # Get all files and directories in the configs directory and its subdirectories
    configs_dir = "configs"
    all_paths = glob.glob(os.path.join(configs_dir, "**/*"), recursive=True)
    all_paths.extend(glob.glob(os.path.join(configs_dir, "*")))  # Include root level items
    
    # Remove thumbnail files and directories
    for path in all_paths:
        if os.path.basename(path).startswith('._'):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Removed thumbnail file: {path}")
                elif os.path.isdir(path):
                    os.rmdir(path)
                    print(f"Removed thumbnail directory: {path}")
            except Exception as e:
                print(f"Error removing {path}: {str(e)}")

if __name__ == "__main__":
    remove_thumbnail_files() 