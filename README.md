Installation

 - In your virtual environment run ```pip install -r requirements.txt```
 - Assumes the following structure
```bash

Data_path_declared_in_geolife_data_utilities
├── Geolife
│   └── Geolife_datasets
```
 - The datasets should be ```.npy``` files
 - The input dataset should be named ```<insert_name_here>X.npy``` and of shape ```(x,15,2)``` where ```x``` is the number of samples. For each sample the input consists of ```15 timesteps``` of ```distance``` (in km) and ```bearing``` (in degrees).
 - Likewise, the output dataset should be named ```<insert_name_here>Y.npy``` and of shape ```(x,2)``` where ```x``` is the number of samples. For each sample the input consists of the output ```distance``` (in km) and ```bearing``` (in degrees).
