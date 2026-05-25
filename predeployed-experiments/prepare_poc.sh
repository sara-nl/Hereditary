echo "Preparing POC environment..."
mkdir -p /tmp/nvflare/poc/example_project/prod_00/server/local/custom
cp -r fs26-demo /tmp/nvflare/poc/example_project/prod_00/server/local/custom/.
echo "Copied fs26-demo to server/local/custom"

# Update authorization.json.default for server and all sites
python3 "$(dirname "$0")/update_auth_permissions.py"
echo "POC is ready to start. Run 'nvflare poc start' to start the server and clients."
