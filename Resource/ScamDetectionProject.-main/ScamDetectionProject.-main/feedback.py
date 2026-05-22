import os
import shutil

# Paths for feedback folders
REAL_FEEDBACK_FOLDER = "new_samples/real"
FAKE_FEEDBACK_FOLDER = "new_samples/fake"

def save_feedback(file_path, correct_label):
    """
    Save audio file to feedback folder based on user-corrected label.
    """
    if correct_label.upper() == "REAL":
        dest_folder = REAL_FEEDBACK_FOLDER
    elif correct_label.upper() == "FAKE":
        dest_folder = FAKE_FEEDBACK_FOLDER
    else:
        print("Invalid label! Must be REAL or FAKE.")
        return

    # Save file with the same name
    base_name = os.path.basename(file_path)
    dest_path = os.path.join(dest_folder, base_name)
    shutil.copy(file_path, dest_path)
    print(f"Saved feedback file to {dest_path}")
