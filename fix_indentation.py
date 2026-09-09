import re

def patch_file():
    path = "services/orchestration/cron_flows.py"
    with open(path, "r") as f:
        content = f.read()

    # Find the commented out line and add 'pass' above it with correct indentation
    pattern = r'(            # conn\.execute\(text\("CALL refresh_continuous_aggregate\(\'rolling_user_stats\', NULL, NULL\);"\)\)  # Temporarily disabled: view does not exist)'
    replacement = r'            pass\n\1'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(path, "w") as f:
        f.write(new_content)
    print("Fixed IndentationError in cron_flows.py")

patch_file()
