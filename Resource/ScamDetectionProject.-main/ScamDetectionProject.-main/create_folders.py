import os

# List of folders to create
folders = ["new_samples/real", "new_samples/fake"]

# Create folders if they don't exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Folders created successfully!")
