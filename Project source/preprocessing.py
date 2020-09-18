import numpy as np
import csv


# 5min - 1153 samples
# 1min - 1441 samples
def load_data_from_csv(sample_id, train_size, test_size, unseen_size, tau, faster_sampling):
    if faster_sampling:
        res = np.zeros(1441)
        dirname = "compiled_data_ssampling/"
    else:
        res = np.zeros(1153)
        dirname = "compiled_data_inc/"

    res_train = np.zeros(shape=(train_size, tau))
    res_test = np.zeros(shape=(test_size, tau))
    res_unseen = np.zeros(shape=(unseen_size, tau))

    i = 0
    with open(dirname + str(sample_id) + '_cpu_usage.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            res[i] = row[2]
            i += 1

    for t in range(0, tau):
        for i in range(0, train_size):
            res_train[i][t] = res[i + t]

    for t in range(0, tau):
        for i in range(0, test_size):
            res_test[i][t] = res[i + train_size + t]

    if faster_sampling:
        i = 0
        with open("compiled_data_ssampling_test/" + str(sample_id) + '_cpu_usage.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                res[i] = row[2]
                i += 1

        for t in range(0, tau):
            for i in range(0, unseen_size):
                res_unseen[i][t] = res[i + t]

    return res_train, res_test, res_unseen
