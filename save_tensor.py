import numpy as np
import csv
import wfdb
import ast
import os
import pickle

path="G:\Works\datamining_project\ecg_classification"
cv_fold = '10'
scpdict = {
    "MI": 0,
    "STTC": 1,
    "CD": 2,
    "HYP": 3
}

with open('ptbxl_database.csv', newline='', mode='r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    x_train = np.empty((1, 12, 1000), dtype=np.float16)
    y_train = np.empty((1, 4), dtype=np.uint8)
    row_index = 0
    for row in reader:
        row_index += 1
        if (row[25] != cv_fold):
            continue
        if(row_index > 50000):
            break

        #One-hot encoding
        this_y = np.array([0, 0, 0, 0])
        this_diag = ast.literal_eval(row[11])
        if(len(this_diag) == 0):
            continue
        for diag in this_diag:
            if not(diag in scpdict):
                continue
            this_y[scpdict[diag]] = 1

        
        data = [wfdb.rdsamp(os.path.join(path, row[26]))]
        data = np.array([signal for signal, meta in data])
        data = np.transpose(data[0]) #12*1000 matrix

        x_train = np.append(x_train, [data], axis = 0)
        y_train = np.append(y_train, [this_y], axis = 0)
        #print(x_train)
        #print(y_train)
    
    x_train = np.delete(x_train, 0 ,0)
    y_train = np.delete(y_train, 0 ,0)
    print(x_train)
    print(y_train)
    with open((cv_fold + '.npy'), 'wb') as saveFile:
        pickle.dump([x_train, y_train], saveFile)

