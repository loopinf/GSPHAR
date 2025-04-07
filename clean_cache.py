import shutil
import os
from pathlib import Path

def clean_cache(clean_type='all'):
    """Clean cache and generated files
    
    Args:
        clean_type (str): Type of cleaning to perform
            'all': Remove all generated files and cache
            'cache': Remove only cache files
            'models': Remove only model checkpoints
            'results': Remove only results files
    """
    # Define directories to clean based on clean_type
    dirs_to_clean = []
    
    if clean_type in ['all', 'cache']:
        dirs_to_clean.extend([
            'cache',
            '__pycache__',
            'config/__pycache__'
        ])
    
    if clean_type in ['all', 'models']:
        dirs_to_clean.extend([
            'checkpoints',
            'models'
        ])
        
    if clean_type in ['all', 'results']:
        dirs_to_clean.extend([
            'results',
            'results/training_history'
        ])
    
    # Remove directories
    for dir_path in dirs_to_clean:
        path = Path(dir_path)
        if path.exists():
            if path.name == '__pycache__':
                # For pycache, also remove in subdirectories
                for cache_dir in Path('.').rglob('__pycache__'):
                    print(f"Removing {cache_dir}")
                    shutil.rmtree(cache_dir)
            else:
                print(f"Removing {path}")
                shutil.rmtree(path)
    
    # Remove .pyc files
    if clean_type in ['all', 'cache']:
        for pyc_file in Path('.').rglob('*.pyc'):
            print(f"Removing {pyc_file}")
            pyc_file.unlink()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean cache and generated files')
    parser.add_argument('--type', type=str, default='all',
                       choices=['all', 'cache', 'models', 'results'],
                       help='Type of cleaning to perform')
    
    args = parser.parse_args()
    
    # Ask for confirmation
    response = input(f"This will remove {args.type} files/directories. Are you sure? (y/n): ")
    if response.lower() == 'y':
        clean_cache(args.type)
        print("Cleaning completed!")
    else:
        print("Operation cancelled.")