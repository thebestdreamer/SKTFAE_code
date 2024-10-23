from skfuzzy import cmeans
import random
import scipy.io
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
#0.56

# fuzzy k-means

def swap(x):
    for i in range(x.shape[0]):
        t = np.argmax(x[i,:])
        if x[i,t]>x[t,t]:
            tem = x[:,i].copy()
            x[:,i] = x[:,t]
            x[:,t] = tem
    return x

def prepocessing_tsne(data, n):
    starttime_tsne = time.time()
    dataset = TSNE(n_components=n, random_state=33).fit_transform(data)
    endtime_tsne = time.time()
    print('cost time by tsne:', endtime_tsne - starttime_tsne)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tsne = scaler.fit_transform(dataset)
    return X_tsne

def accuracy(num_classes,cluster_labels,label):
    dic_pred = np.zeros([num_classes, num_classes])
    for i in range(len(cluster_labels)):
        dic_pred[np.int(label[i]), cluster_labels[i]] += 1

    dic_pred = swap(dic_pred)
    # print(dic_pred)
    acc = np.zeros([dic_pred.shape[0], 1])
    sum_num = 0
    for i in range(dic_pred.shape[0]):
        acc[i, 0] = dic_pred[i, i] / int(np.sum(dic_pred[:, i]))
        sum_num += dic_pred[i, i]
        # print("第{:d}类正确率:{:.2f}".format(i, dic_pred[i, i] / 1000))
    print('总体正确率{:.2f}'.format(sum_num / dic_pred.sum()))

    return accuracy

def avg(data,place):
    dis = np.zeros([place.shape[0],place.shape[0]])
    for i,itemi in enumerate(place):
        for j in range(i,len(place)):
            dis[i,j] = np.sum((data[itemi[0],:]-data[place[j][0]])**2)
    if i == 1:
        return 0
    else:
        return np.sum(dis)/len(place)/(len(place)-1+0.01)/2


def db_para(data,y_pre,center):
    c = len(center)
    dis_a = []
    dis_c = np.zeros([c,c])
    center = np.array(center)
    for i in range(c):
        place = np.argwhere(y_pre==i)
        dis_a.append(avg(data,place))
        a=0
    for i in range(c):
        for j in range(c):
            dis_c[i,j] = np.sum((center[i,:]-center[j,:])**2)
    db = 0
    for i in range(c):
        tem_a = dis_a.copy()
        tem_c = dis_c[i,:].copy()
        tem_a.remove(dis_a[i])
        tem_c = np.delete(tem_c,i)
        db += max((dis_a[i]+tem_a)/tem_c)

    db = db/c
    return db

np.random.seed(42)
random.seed(42)

def fcm(fea_name, num_classes=4,mask_ratio=0.75,patch_size=16):
    np.random.seed(42)

    data = np.load('fea_label/'+fea_name)
    label = np.load('fea_label/label.npy')

    print(data.shape)

    # mean = data.mean(0)
    # std = data.std(0)
    # for i in range(0, data.shape[0]):
    #     data[i, :] = (data[i, :] - mean) / std

    pca = PCA(2)
    data_pca = pca.fit_transform(data[:, :])

    # Maximum number of iterations
    MAX_ITER = 200

    # Number of data points
    n = data.shape[0]

    # Fuzzy parameter
    t = np.linspace(0.8, 1.5, 40)
    t = list(t)
    # t = [1.3]
    m = 1.3
    for m in t:
        np.random.seed(42)
        random.seed(42)

        labels = cmeans(data, c=num_classes, m=m, error=0.001, maxiter=300)
        print("自带评价指标:", labels[-1])
        labels = labels[0]
        labels = labels.argmax(axis=0)
        labels = list(labels)
        print("当前超参数{:.2f}".format(m))

        # sklearn自带算法  DBI的值最小是0，值越小，代表聚类效果越好。
        cluster_score_DBI = metrics.davies_bouldin_score(data, labels)
        cluster_score_real_DBI = metrics.davies_bouldin_score(data, label[:])
        cluster_score_NMI = metrics.normalized_mutual_info_score(label[:], labels)
        cluster_score_F = metrics.f1_score(label[:], labels, average='micro')
        print("cluster_score_DBI:", cluster_score_DBI)
        print('cluster_score_real_DBI', cluster_score_real_DBI)
        print("cluster_score_NMI:", cluster_score_NMI)
        print('cluster_score_F:', cluster_score_F)

        a = accuracy(num_classes=num_classes, cluster_labels=labels, label=label)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=label[:])
    plt.show()

    data_t_sne = prepocessing_tsne(data, 2)
    plt.scatter(data_t_sne[:, 0], data_t_sne[:, 1], c=label)
    plt.savefig('Tsne_img/'+'cls'+str(num_classes)+'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + 'acc'+ str(a)[:5] + '.png')
    plt.show()
