import glob
import os

def check_library_usage(file_path):
    uses_monai = False
    uses_torch = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Check for import statements
            uses_monai = 'monai' in content and ('import monai' in content or 'from monai' in content)
            uses_torch = 'torch' in content and ('import torch' in content or 'from torch' in content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return uses_monai, uses_torch

def main(file_pattern):
    # Find all best_solution.py files in log/*-*/ directories
    files = glob.glob(file_pattern)
    
    if not files:
        print("No files found matching the pattern 'log/*-*/best_solution.py'")
        return
    
    # Track results
    results = []
    
    for file_path in files:
        monai, torch = check_library_usage(file_path)
        status = []
        if monai:
            status.append("monai")
        if torch:
            status.append("torch")
        if not status:
            status.append("neither")
        
        results.append((file_path, ", ".join(status)))
    results = sorted(results, key=lambda x : int(x[0].replace("logs/", "").split("-")[0]))
    
    # Print results
    print("\nLibrary usage in best_solution.py files:")
    print("-" * 50)
    for file_path, status in results:
        print(f"{file_path}: {status}")

if __name__ == "__main__":
    main("logs/*-*/best_solution.py")