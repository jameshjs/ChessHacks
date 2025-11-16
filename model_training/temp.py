def _read_file_content(filepath):
    """Read file content for embedding in image."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
        print("opened file \n")
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

# Read file contents at module load time (when running locally)
_model_py = _read_file_content("./model.py")
_dataset_py = _read_file_content("./dataset.py")
_position_encoder_py = _read_file_content("./position_encoder.py")

