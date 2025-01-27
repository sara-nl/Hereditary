# Hereditary workshop 3
This branch contains the code for the workshop 3 of the Hereditary project. The readme will first explain how to download the data, then how to run the experiments. 
## Data download instructions
### CLEF
In order to download the CLEF data, you need to follow these steps:
1. Sign into the grouppage at https://hereditary.dei.unipd.it/groupoffice/#summary 
2. Go to files -> Hereditary/Data/BRAINTEASER_ALS_MS_datasets and download the prospective.zip and retrospective.zip files. 
3. Extract the zip files  

### FETS
In order to download the FETS data, you need to follow these steps: 
1. Create a synapse.org account 
2. Go to your account settings (https://accounts.synapse.org/authenticated/myaccount?appId=synapse.org) and create a Personal Access Token (and make sure to save it) 
3. Sign up for FETS 2024 here: https://www.synapse.org/Synapse:syn54079892/wiki/626854 
4. Complete the data access form    
5. For downloading using Python/CLI, install this package: https://pypi.org/project/synapseclient/ 
6. Go to the files page: https://www.synapse.org/Synapse:syn29264504 
7. Select the right task and add the files to your cart.  
8. View your download list: https://www.synapse.org/DownloadCart:0 and make sure it’s correct 
9. Call this command in your terminal to download the files: synapse get-download-list 
10. Enter your username and personal access token when prompted and the download will start. 


## Federated learning with XGBoost and CLEF
See the xgboost-CLEF directory for the code to partition the data and run the federated learning experiments. 


## Federated learning with FETS
WIP

## Server setup
During the workshop we will use a server hosted on the SURF research cloud as the SuperLink for our experiments. 
When creating this server (type ubuntu 2204 sudo enabled), follow these steps:
```
sudo apt update
sudo apt install python3.11
cd data/VOLUME_NAME
python3.11 -m venv flwr-venv
source flwr-venv/bin/activate
pip install flwr==1.14.0
```

After following these steps, you should be able to start the superlink, for the specific command see the readme of any experiment in this branch. 

