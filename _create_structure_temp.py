import os

def create_project_structure():
    # Use the current directory as the base
    base_dir = "." 
    print(f"Using base directory: {os.path.abspath(base_dir)}")

    directories = [
        "LICENSE", # Changed: This is usually a file, but the script treats it as a dir. Will create dir.
        "Makefile", # Changed: This is usually a file, but the script treats it as a dir. Will create dir.
        "README.md", # Changed: This is usually a file, but the script treats it as a dir. Will create dir.
        "data/external",
        "data/interim",
        "data/processed",
        "data/raw",
        "docs",
        "models", # Note: This 'models' dir is for saved model files, different from src/models
        "notebooks",
        "references",
        "reports/figures",
        "src/data",
        "src/features",
        "src/models", # Note: This 'src/models' dir is for model definition code
        "src/visualization",
    ]

    files = [
        "requirements.txt",
        "setup.py",
        "tox.ini",
        "src/__init__.py",
        "src/data/make_dataset.py",
        "src/features/build_features.py",
        "src/models/predict_model.py",
        "src/models/train_model.py",
        "src/visualization/visualize.py",
    ]

    # Correctly handle files treated as directories in the original list
    special_files = ["LICENSE", "Makefile", "README.md"]
    actual_directories = [d for d in directories if d not in special_files]

    for directory in actual_directories:
        dir_path = os.path.join(base_dir, directory)
        # Prevent creating '.' directory itself
        if dir_path == ".":
             continue
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Combine special files with other files
    all_files = files + special_files

    for file in all_files:
        file_path = os.path.join(base_dir, file)
        # Create parent directory if it doesn't exist (e.g., for src/data/make_dataset.py)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Ensure we don't try to create a file "."
        if file_path == ".":
            continue
        # Create empty file
        with open(file_path, 'w') as f:
            f.write("") 
        print(f"Created file: {file_path}")

create_project_structure() 