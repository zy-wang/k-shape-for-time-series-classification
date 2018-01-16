import numpy as np
from numpy import genfromtxt
from kshape import kshape, zscore
import time
from sklearn.metrics import confusion_matrix

begin = time.clock()

def load_dataset(dataset_name, dataset_folder):
    #train_file_path = dataset_folder + "/" + dataset_name + "/" + dataset_name + "_TRAIN"
    train_file_path = dataset_name + "/" + dataset_name + "_TRAIN2"
    #train_file_path = dataset_name + "/" + dataset_name + "_train5.txt"
    #test_file_path = dataset_folder + "/" + dataset_name + "/" + dataset_name + "_TEST"
    test_file_path = dataset_name + "/" + dataset_name + "_TEST"
    #test_file_path = dataset_name + "/" + dataset_name + "_test5.txt"
    # read training data into numpy arrays
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_labels = train_raw_arr[:, 0]
    train_data = train_raw_arr[:, 1:]
    #print(train_data[6][23:59])
    # read test data into numpy arrays
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_labels = test_raw_arr[:, 0]
    test_data = test_raw_arr[:, 1:]
    return train_data, train_labels, test_data, test_labels

train_data, train_label, test_data, test_label = load_dataset('ECGFiveDays', 'data')
#test_data, test_label, train_data, train_label = load_dataset('Gun_Point', 'data')
train_data = list(train_data)
cluster_data = []

L = len(train_data[0])
seq_len = 7
def Segmentation1(train_data, train_label):
    a = []
    sep = 5
    l = int((L-seq_len)/sep) + 1
    for i in range(len(train_data)):
        for j in range(l):
            a = train_data[i][j*sep:j*sep+seq_len]
            a = list(a)
            a.append(train_label[i])
            cluster_data.append(a)

Segmentation1(train_data, train_label)         
cluster_data = np.array(cluster_data)

X = cluster_data[:, :-1]#切分后的数据
Y = cluster_data[:, -1]#切分后数据的标签

n_clusters = 5

#cluster = kshape(zscore(X), n_clusters)

def get_cluster_result(cluster, n_clusters):
    cluster_result = []
    for i in range(n_clusters):
        tmp = []
        if len(cluster[i][1]) != 0:
            tmp = cluster[i][1]
            cluster_result.append(tmp)
    return cluster_result

#cluster_result = get_cluster_result(cluster, n_clusters)
#real_numOfcluster = len(cluster_result)

def Categorized(Y, cluster_result, real_numOfcluster):
    d = []
    for i in range(real_numOfcluster):
        c = []
        for j in range(len(cluster_result[i])):
            c.append(Y[cluster_result[i][j]])
        d.append(c)
    return d

#d = Categorized(Y, cluster_result)

def return_shapelets(real_numOfcluster, d, cluster_result):
    shapelets_vec = []
    rate = []
    for i in range(real_numOfcluster):
        m = 0
        n = 0
        for j in range(len(d[i])):
            if d[i][j] == 1:
                m = m+1
            else:
                n = n+1
        rate.append(m/(m+n))
    min_num = min(rate) #2占多数
    max_num = max(rate) #1占多数
    index1 = rate.index(min_num)
    index2 = rate.index(max_num)
    l1 = len(d[index1])
    l2 = len(d[index2])
    num_1 = 0 #1片段数量
    num_2 = 0 #2片段数量
    total_shape1 = [0]*seq_len
    total_shape2 = [0]*seq_len
    for i in range(l1):
        if d[index1][i] == 2:
            num_2 = num_2 +1
            total_shape2 = total_shape2 + X[cluster_result[index1][i]]
    shapelets2 = np.array(total_shape2) / num_2
    shapelets_vec.append(shapelets2) #用来判定2的

    for i in range(l2):
        if d[index2][i] == 1:
            num_1 = num_1 + 1
            total_shape1 = total_shape1 + X[cluster_result[index2][i]]
    shapelets1 = np.array(total_shape1) / num_1
    shapelets_vec.append(shapelets1) #用来判定1的

    return shapelets_vec

#shapelets_vec = return_shapelets(real_numOfcluster)

def find_min_distance(time_series, shapelets):
    time_series = np.array(time_series)
    L = len(time_series)
    l = len(shapelets)
    dist_list = []
    for i in range(L-l+1):
        dist = np.linalg.norm(time_series[i : i+l] - shapelets)
        dist_list.append(dist)
    min_dist = min(dist_list)
    return min_dist


def test(shapelets_vec, test_data):
    predict_label = []
    label = 1
    for i in range(len(test_data)):
        dist_vec = []
        for j in range(2):
            dist = find_min_distance(test_data[i], shapelets_vec[j])
            dist_vec.append(dist)
        if dist_vec[0] < dist_vec[1]:
            label = 2
        else:
            label = 1
        predict_label.append(label)

    return predict_label


def return_best_shapelets(maxIter, minNum):
    shapelets_tmp = []
    for i in range(maxIter):
        cluster = kshape(zscore(X), n_clusters)
        cluster_result = get_cluster_result(cluster, n_clusters)
        real_numOfcluster = len(cluster_result)
        d = Categorized(Y, cluster_result, real_numOfcluster)
        shapelets_vec = return_shapelets(real_numOfcluster, d, cluster_result)
        predict_label = test(shapelets_vec,train_data)
        FP_FN = confusion_matrix(train_label, predict_label)[0][1]+confusion_matrix(train_label,predict_label)[1][0]
        print( FP_FN)
        if FP_FN<minNum :
            minNum = FP_FN
            shapelets_tmp = shapelets_vec
    return shapelets_tmp, minNum
shapelets_tmp, minNum = return_best_shapelets(20,1000)
#predict_label = test(shapelets_vec)
#print(confusion_matrix(test_label, predict_label))