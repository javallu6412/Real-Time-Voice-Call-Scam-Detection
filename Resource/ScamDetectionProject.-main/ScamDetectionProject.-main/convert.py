import os

# உங்கள் தற்போதைய directory-ல் உள்ள folders
print("Current folder:", os.getcwd())
print("Subfolders:", os.listdir())

# real folder-ல் உள்ள கோப்புகள்
if os.path.exists("real"):
    print("\nFiles in 'real' folder:", os.listdir("real"))
else:
    print("\n'real' folder இல்லை!")

# fake folder-ல் உள்ள கோப்புகள்
if os.path.exists("fake"):
    print("\nFiles in 'fake' folder:", os.listdir("fake"))
else:
    print("\n'fake' folder இல்லை!")