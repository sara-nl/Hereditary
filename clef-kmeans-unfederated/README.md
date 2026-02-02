# Centralized K-means on CLEF data
This folder contains the unfederated K-means implementation for the CLEF data and example scripts

If you would like to run the code, download the data [here](https://mega.nz/file/05wQlACa#IWL1MDrAQF1MJTBtSZLpn_ziE7e5EPbW81RUz2j4LoQ), extract it and set the CLEF_DATA_PATH_UNFEDERATED environment variable to the path of the extracted data like so:

```
export CLEF_DATA_PATH_UNFEDERATED="/path/to/extracted/data/retrospective/ALS/CSV/data/datasetC/"
```

## Installation 
It is recommended to make a python 3.11 venv like so:
```
python3.11 -m venv kmeans_venv
source kmeans_venv/bin/activate
pip install -r requirements.txt
```
As the plot_data.py script also opens an interactive data viewer in the webbrowser, this might not work well in a docker image, but this has not been tested. 


## running the code
`kmeans.py` runs KMeans on the CLEF data and will display the final cluster centers. The data is preprocessed in the same way as in the federated codebase. 
`plot_data.py` shows the output of different dimensionality reduction algorithms on the data, as PCA doesn't show an impressive separation. This can be used as an inspritation for the hackathon. This includes interactive output that will be openened in the browser. 

## Example scripts
`example_kmeans_vis.py` shows a plot for each iteration of KMeans, the output will also be presented in the workshop. 
`example_elbow_method` shows how the elbow method can be used to determine the optimal amount of clusters for KMeans, the output will also be shown and explained in the workshop. 