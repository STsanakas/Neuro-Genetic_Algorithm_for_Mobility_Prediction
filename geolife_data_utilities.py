import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
def get_data(mode):
  data_path = get_geolife_path()
  X = np.delete(np.load(data_path+'Geolife_datasets/'+mode+'X.npy'), 0, 0)
  Y = np.delete(np.load(data_path+'Geolife_datasets/'+mode+'Y.npy'), 0, 0)
  X=X*[1,1/360]
  Y=Y*[1,1/360]
  trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.33, random_state=42)
  return trainX, testX, trainY, testY

def get_data_path():
  return '/insert/data/path/'

def get_geolife_path():
  return get_data_path()+'Geolife/'

def get_code_path():
  return get_data_path()+'Federated Mobility/'

#Splits the map in 4 parts and returns the part that given point belongs to
def area(lat, lon, avg_lat=34.8220095, avg_lon=104.1440295):
  if lon>avg_lon:
    if lat>avg_lat:
      return 'northeast'
    else:
      return 'southeast'
  else:
    if lat>avg_lat:
      return 'northwest'
    else:
      return 'southwest'