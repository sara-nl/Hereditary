import json
import glob
import sys

# Default permission template for roles with "any"
DEFAULT_PERMISSIONS = {
    "submit_job": "any",
    "clone_job": "none",
    "manage_job": "o:submitter",
    "download_job": "o:submitter",
    "view": "any",
    "operate": "o:site",
    "shell_commands": "o:site",
    "byoc": "none",
    "server-predeployed-flwr": "any",
    "get_job_log": "any"
}

def process_auth_file(filepath):
    print(f"Updating permissions in: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    permissions = data.get("permissions", {})

    for role, role_perms in permissions.items():
        # If role is "any", expand it to default permissions
        if role_perms == "any":
            permissions[role] = DEFAULT_PERMISSIONS.copy()
        # If role is already a dict, leave it unchanged

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

# Determine which file to process
if len(sys.argv) > 1:
    filepath = sys.argv[1]
    process_auth_file(filepath)
else:
    # Default: find authorization.json.default files in server and all sites (site-*)
    auth_files = glob.glob("/tmp/nvflare/poc/example_project/prod_00/*/local/authorization.json.default")
    for filepath in auth_files:
        process_auth_file(filepath)

print("Permissions update completed successfully for all roles and sites.")
