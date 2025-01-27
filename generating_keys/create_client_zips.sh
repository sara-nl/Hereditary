#!/bin/bash
# Script to package the client credentials into a zip file to send to participants who will host a supernode
# example usage: ./create_client_zips.sh 20
# Create client_zips directory if it doesn't exist
mkdir -p client_zips

# Check if CA certificate exists
if [ ! -f "certificates/ca.crt" ]; then
    echo "Error: CA certificate not found at certificates/ca.crt"
    exit 1
fi

# Create zip for each client
for i in $(seq 1 $1); do
    # Check if client credentials exist
    if [ ! -f "keys/client_credentials_$i" ] || [ ! -f "keys/client_credentials_$i.pub" ]; then
        echo "Warning: Client credentials $i not found, skipping..."
        continue
    fi
    
    # Create zip file
    zip -j "client_zips/client_${i}_credentials.zip" \
        "certificates/ca.crt" \
        "keys/client_credentials_$i" \
        "keys/client_credentials_$i.pub"
        
    echo "Created client_zips/client_${i}_credentials.zip"
done

echo -e "\nDone! Client credential zips have been created in the client_zips directory" 