import math
import numpy as np
import keras.backend as K
import tensorflow as tf

def calculate_initial_compass_bearing(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def convert_to_distance_bearing(pointAlat, pointAlon, pointBlat, pointBlon):
	pointA=(pointAlat,pointAlon)
	pointB=(pointBlat,pointBlon)
	A=calculate_initial_compass_bearing(pointA, pointB)
	from math import sin, cos, sqrt, atan2, radians	
	R = 6373.0
	lat1 = radians(pointAlat)
	lon1 = radians(pointAlon)
	lat2 = radians(pointBlat)
	lon2 = radians(pointBlon)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	distance = R * c
	return distance, A

def create_timeseries(mydata, in_s=15, out_s=4, disbear=True):
  #Boolean disbear False creates a timeseries without transforming to distance/bearing
  datasetX=[]
  datasetY=[]
  new_data=[]
  new_dataY=[]
  for i in range(mydata.shape[0]-in_s-out_s):
    if disbear:
      new_data.append(convert_to_distance_bearing(mydata[i,0], mydata[i,1], mydata[i+1,0], mydata[i+1,1]))
      new_dataY.append(convert_to_distance_bearing(mydata[i+in_s-1,0], mydata[i+in_s-1,1], mydata[i+in_s+out_s,0], mydata[i+in_s+out_s,1]))
    else:
      new_data.append((mydata[i,0], mydata[i,1]))
      new_dataY.append((mydata[i+in_s-1,0], mydata[i+in_s+out_s,0]))
  mydata=np.array(new_data)
  mydataY=np.array(new_dataY)
  for i in range(mydata.shape[0]-in_s):    
    input=mydata[i:i+in_s]
    output=mydataY[i]
    datasetX.append(input)
    datasetY.append(output)
  np.array(datasetX)
  return np.array(datasetX), np.array(datasetY)

def cartesian_error(y_true, y_pred):
  x1 = y_true[:,0]*K.cos(0.017453292519943295*(90-(y_true[:,1]*360)))
  y1 = y_true[:,0]*K.sin(0.017453292519943295*(90-(y_true[:,1]*360)))
  x2 = y_pred[:,0]*K.cos(0.017453292519943295*(90-(y_pred[:,1]*360)))
  y2 = y_pred[:,0]*K.sin(0.017453292519943295*(90-(y_pred[:,1]*360)))     
  loss = 1000*K.sqrt((x2-x1)**2+(y2-y1)**2)
  nan_value = tf.ones_like(loss)*5000  
  loss=K.switch(tf.math.is_nan(loss), nan_value, loss)
  loss=K.switch(tf.math.is_inf(loss), nan_value, loss)
  return loss


def cartesian_eval(y_true, y_pred):

  x1 = y_true[:,0]*np.cos(0.017453292519943295*(90-(y_true[:,1]*360)))
  y1 = y_true[:,0]*np.sin(0.017453292519943295*(90-(y_true[:,1]*360)))
  x2 = y_pred[:,0]*np.cos(0.017453292519943295*(90-(y_pred[:,1]*360)))
  y2 = y_pred[:,0]*np.sin(0.017453292519943295*(90-(y_pred[:,1]*360)))     
  loss = 1000*np.sqrt((x2-x1)**2+(y2-y1)**2)  
  return round(np.average(loss),2)