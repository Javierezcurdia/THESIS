import os
import shutil
import random
import http.client
import json
import numpy as np
import pandas as pd
import ssl

conn = http.client.HTTPSConnection("tracker.elioslab.net", context=ssl._create_unverified_context())
## conn = http.client.HTTPSConnection("tracker.elioslab.net")
payload = json.dumps({
  "username": "LaFauci",
  "password": "LF_Password",
  "tenant": "sci-tenant"
})
headers = {
  'Content-Type': 'application/json'
}
conn.request("POST", "/v1/login", payload, headers)
res = conn.getresponse()
data = res.read()
data=json.loads(data.decode("utf-8"))
token=data['token']
##print  ("Token: ", token)

def get_timeseries_from_measurement(measureName):
  payload = ''
  headers = {
    'Content-Type': 'application/json',
    'Authorization': token
  }
  conn.request("GET", "/v1/measurements/"+measureName+"/timeserie?limit=2000&sort=%7B%20%22timestamp%22:%20%22asc%22%20%7D", payload, headers)
  res = conn.getresponse()
  response = res.read()
  response=json.loads(response)
  ##print(response)
  timeseries_signal=response['docs']
  return timeseries_signal

def extracting_signals(timeseries_signal, label):
    values_all = []
    for k in timeseries_signal:
        values = [k["values"][0], k["values"][1], k["values"][2], k["values"][3], k["values"][4], k["values"][5], label]
        values_all.append(values)

    ##values_all=values_all[0][10:-10]
    signal_all = np.array(values_all)
    return signal_all

def extracting_testing_signals(timeseries_signal,label,first_sample,last_sample):
    values_all = []
    for i in range(first_sample, last_sample):
        k = timeseries_signal[i]
        values = [k["values"][0], k["values"][1], k["values"][2], k["values"][3], k["values"][4], k["values"][5], label]
        # Append the extracted values to the values_all list
        values_all.append(values)
    # Convert the list of values into a numpy array
    signal_all = np.array(values_all)
    return signal_all

# Create folders if they don't exist
if not os.path.exists("./testing"):
    os.makedirs("./testing")
if not os.path.exists("./training"):
    os.makedirs("./training")
if not os.path.exists("./validation"):
    os.makedirs("./validation")

file_names_testing = ["test-All","test2-All","test3-All"]
file_names_validation = [["test2-pistaPancaniLM","test3-pistaPancaniLM","test2-pistaBluLM"],["test7-pistaLuciaEN","test8-pistaLuciaEN","test9-pistaLuciaEN"],["test10-pistaLuciaENsp","test11-pistaLuciaENsp"],["test2-pistaAlpettaLMuo"]]
file_names_training = [["test1-pistaAlpettaLM","test2-pistaAlpettaLM","test3_pistaAlpettaLM","test-pistaAlpettaneraLM","test1-pistaPancaniLM"],["test3-pista1","test4-pistababyEN","test5-pista1EN","test5-pistababyEN","test6-pista1EN","test6-pistaLuciaENsp"],["test1-pista1ENsp","test3-pistaLuciaENsp","test4-pistaLuciaENsp","test5-pistaLuciaENsp","test8-pistaLuciaENsp","test9-pistaLuciaENsp"],["test-pistaPancaniLMuo","test-pistaBluLMuo"]]

for label, files in enumerate(file_names_training, start=0): 
    for file in files:
        timeseries_signal = get_timeseries_from_measurement(file)
        signal_all = extracting_signals(timeseries_signal,label)
        signal_file_name = file.split('.')[0] + '.txt'
        with open(os.path.join("./training", signal_file_name), 'w') as f:
            for row in signal_all:
                f.write(' '.join(map(str, row)) + '\n')
            f.flush()

for label, files in enumerate(file_names_validation, start=0): 
    for file in files:
        timeseries_signal = get_timeseries_from_measurement(file)
        signal_all = extracting_signals(timeseries_signal,label)
        signal_file_name = file.split('.')[0] + '.txt'
        with open(os.path.join("./validation", signal_file_name), 'w') as f:
            for row in signal_all:
                f.write(' '.join(map(str, row)) + '\n')
            f.flush()

for file in file_names_testing:
    timeseries_signal = get_timeseries_from_measurement(file)
    
    if file=="test-All":
        signal_all1 = extracting_testing_signals(timeseries_signal,1,10,330)
        signal_all2 = extracting_testing_signals(timeseries_signal,2,330,410)
        signal_all3 = extracting_testing_signals(timeseries_signal,0,432,632)
        signal_all4 = extracting_testing_signals(timeseries_signal,3,626,len(timeseries_signal)-11)
        signal_all = np.concatenate((signal_all1, signal_all2, signal_all3, signal_all4), axis=0)
        ##print(len(timeseries_signal))
        signal_file_name = file.split('.')[0] + '.txt'
        with open(os.path.join("./testing", signal_file_name), 'w') as f:
            for row in signal_all:
                f.write(' '.join(map(str, row)) + '\n')
            f.flush()
            
    if file=="test2-All":
        signal_all1 = extracting_testing_signals(timeseries_signal,0,10,198)
        signal_all2 = extracting_testing_signals(timeseries_signal,3,198,292)
        signal_all3 = extracting_testing_signals(timeseries_signal,1,292,493)
        signal_all4 = extracting_testing_signals(timeseries_signal,2,505,len(timeseries_signal)-11)
        signal_all = np.concatenate((signal_all1, signal_all2, signal_all3, signal_all4), axis=0)
        ##print(len(timeseries_signal))
        signal_file_name = file.split('.')[0] + '.txt'
        with open(os.path.join("./testing", signal_file_name), 'w') as f:
            for row in signal_all:
                f.write(' '.join(map(str, row)) + '\n')
            f.flush()
           
    if file=="test3-All":
        signal_all1 = extracting_testing_signals(timeseries_signal,2,10,147)
        signal_all2 = extracting_testing_signals(timeseries_signal,3,147,268)
        signal_all3 = extracting_testing_signals(timeseries_signal,0,268,388)
        signal_all4 = extracting_testing_signals(timeseries_signal,1,393,len(timeseries_signal)-1)
        signal_all = np.concatenate((signal_all1, signal_all2, signal_all3, signal_all4), axis=0)
        ##print(len(timeseries_signal))
        signal_file_name = file.split('.')[0] + '.txt'
        with open(os.path.join("./testing", signal_file_name), 'w') as f:
            for row in signal_all:
                f.write(' '.join(map(str, row)) + '\n')
            f.flush()

file_training_names = [file for file in os.listdir("./training")]
file_validation_names = [file for file in os.listdir("./validation")]
file_testing_names = [file for file in os.listdir("./testing")]

def preprocess_file(directory, file_name):
    file_path = os.path.join(directory, file_name)
    column_names = ["xAcc", "yAcc", "zAcc", "xGyr", "yGyr", "zGyr", "label"]
    df = pd.read_table(file_path, delimiter=" ", names=column_names)  # Use names directly for header info
    return df

training_samples = []
training_dataset = []
for file_name in file_training_names:
    df=preprocess_file("./training", file_name)
    training_samples.append(df)
training_dataset = pd.concat(training_samples, ignore_index=True)

validation_samples = []
validation_dataset=[]
for file_name in file_validation_names:
    df = preprocess_file("./validation", file_name)
    validation_samples.append(df)
validation_dataset = pd.concat(validation_samples, ignore_index=True)

testing_samples = []
testing_dataset = []
for file_name in file_testing_names:
    df = preprocess_file("./testing", file_name)
    testing_samples.append(df)
testing_dataset = pd.concat(testing_samples, ignore_index=True)
 
from sklearn.preprocessing import StandardScaler
def scaling(dataset):
    # Select all columns except the last one for scaling
    X = dataset.iloc[:, :-1]
 
    # Initialize the scaler
    scaler = StandardScaler()
 
    # Scale the selected columns
    X_scaled = scaler.fit_transform(X)
 
    # Convert the scaled array back to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=dataset.columns[:-1])
 
    # Concatenate the scaled columns with the last column
    final_df = pd.concat([X_scaled_df, dataset.iloc[:, -1].reset_index(drop=True)], axis=1)

    return final_df

dataset_train=scaling(training_dataset)
dataset_val=scaling(validation_dataset)
dataset_test=scaling(testing_dataset)

def windowing(dataset, length):
    datasets = [dataset[i:i+length] for i in range(0, len(dataset) - length, length)]
    return datasets

dataset_test = windowing(dataset_test, 15)

def sliding_windowing(dataset, length, step):
    datasets = [dataset[i:i+length] for i in range(0, len(dataset)-length+1, step)]
    return datasets

step = 5  # Adjust the overlap as needed
dataset_train = sliding_windowing(dataset_train, 15, step)
dataset_val = sliding_windowing(dataset_val, 15, step)


##print("dataset_train after windowing:",dataset_train)
##print("dataset_test after windowing:",dataset_test)

def get_labels(dataset):
    labels = []
    for window in dataset:
        label_counts = window['label'].value_counts()
        most_common_label = label_counts.idxmax()
        ##print(most_common_label)
        labels.append(most_common_label)
    return labels

labels_train = get_labels(dataset_train)
labels_val = get_labels(dataset_val)
labels_test = get_labels(dataset_test)

def droplbl(dataset):
    dataset_nolbl = [block.drop("label", axis=1) for block in dataset]
    return dataset_nolbl
# Usage of droplbl function
dataset_train = droplbl(dataset_train)
dataset_val = droplbl(dataset_val)
dataset_test = droplbl(dataset_test)

X_train=dataset_train
y_train=labels_train
X_val=dataset_val
y_val=labels_val
X_test=dataset_test
y_test=labels_test

for item in X_test:
     print(len(item))

import numpy as np
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)




