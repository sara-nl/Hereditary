# Generating keys and certificates
This directory contains the scripts needed to generate the keys and certificates to let the supernodes and superlink communicate securely and only allow authorized supernodes to connect to the superlink.

### Encrypt the traffic
To create a self-signed TLS certificate, you can run the following command in the `generating_keys` directory:
```bash
bash generate_cert.sh
```
The keys will be stored in the a directory called `certificates`.

### Create authentication keys
Then, to create authentication keys for the supernodes, you can run the following command in the `generating_keys` directory:
```bash
bash generate_auth_keys.sh 20
```
This will create 20 authentication keys, which will be used to authenticate the supernodes to the superlink, stored in the `keys` directory.

You will then need to distribute the following files to the owners of the supernodes:
* `certificates/ca.crt`: the CA certificate
* `keys/client_credentials_1.pub`: the public key of the supernode
* `keys/client_credentials_1`: the private key of the supernode

The superlink will need the following files:
* `certificates/ca.crt`: the CA certificate
* `certificates/server.pem`: the server certificate
* `certificates/server.key`: the server private key
* `keys/client_public_keys.csv`: the list of public keys of the authorized supernodes
* `keys/server_credentials`: the server private key
* `keys/server_credentials.pub`: the server public key

### Package the keys and certificates
To create zipfiles containing a single set of keys and the root certificate, you can use the `create_client_zips.sh` script.
```bash
bash create_client_zips.sh 20
```
This will create 20 zipfiles, each containing the keys and the root certificate, assuming 20 keys were generated. The zip files can then be distributed to the owners of the supernodes.

### resources:
* https://flower.ai/docs/framework/how-to-enable-tls-connections.html
* https://flower.ai/docs/examples/flower-authentication.html
* https://flower.ai/docs/framework/how-to-authenticate-supernodes.html


