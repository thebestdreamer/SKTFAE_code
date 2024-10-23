import os.path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as cluster
from seaborn import heatmap

def avg(data,place):
    dis = np.zeros([place.shape[0],place.shape[0]])
    for i,itemi in enumerate(place):
        for j in range(i,len(place)):
            dis[i,j] = np.sum((data[itemi[0],:]-data[place[j][0]])**2)
    if i == 1:
        return 0
    else:
        return np.sum(dis)/len(place)/(len(place)-1+0.01)/2



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

def compute_optimal_transport(M, r, c, lam, epsilon=1e-8):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))#行归r化，注意python中*号含义
        P *= (c / P.sum(0)).reshape((1, -1))#列归c化
    return P, np.sum(P * M)

def find_center(x,mask_ratio,patch_size,num_cluster):
    clu = cluster.KMeans(n_clusters=num_cluster,random_state=42)
    tem = x
    y_pred = clu.fit_predict(tem)
    center_list = []
    for i in range(num_cluster):
        center_list.append(tem[np.argwhere(y_pred==i)].mean(axis=0))
    return np.array(center_list)

def load_center(x,mask_ratio,patch_size,num_cluster):
    center = np.load(str(num_cluster)+'fea_'+x+'_label/'+ 'center' + '_fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy')
    return center


def sk_cluster(fea_name,num_classes,mask_ratio,patch_size,label_name,center_load = False,need_image=False):
    np.random.seed(3407)
    type = fea_name.split('_')[1]
    data = np.load(fea_name)
    label = np.load(label_name)
    if center_load:
        center = load_center(type,mask_ratio,patch_size,num_classes)
    else:
        center = find_center(data,mask_ratio,patch_size,num_classes)
    print(data.shape)
    d = np.zeros([data.shape[0],num_classes])

    for pl_i, i in enumerate(data):
        for pl_j, j in enumerate(center):
            d[pl_i, pl_j] = np.mean((i - j) ** 2)
    num, cls = d.shape[:]
    t, v = compute_optimal_transport(d, (np.ones([num, ]) / num), (np.ones([cls, ]) / cls),20)
    for i, _ in enumerate(t):
        t[i, :] /= t[i, :].sum()
    pseudo_label = np.argmax(t, axis=1)
    y_ls = list(pseudo_label)
    dic_pred = np.zeros([num_classes, num_classes])
    for i in range(len(y_ls)):
        dic_pred[np.int32(label[i]), pseudo_label[i]] += 1

    # # sklearn自带算法  DBI的值最小是0，值越小，代表聚类效果越好。
    # cluster_score_DBI = metrics.davies_bouldin_score(data, pseudo_label)
    # cluster_score_real_DBI = metrics.davies_bouldin_score(data, label)
    # cluster_score_NMI = metrics.normalized_mutual_info_score(label, pseudo_label)
    # cluster_score_F = metrics.f1_score(label, pseudo_label, average='micro')
    # print("cluster_score_DBI:", cluster_score_DBI)
    # print('cluster_score_real_DBI', cluster_score_real_DBI)
    # print("cluster_score_NMI:", cluster_score_NMI)
    # print('cluster_score_F:', cluster_score_F)

    # print(dic_pred)
    dic_pred = swap(dic_pred)
    print(dic_pred)
    acc = np.zeros([dic_pred.shape[0], 1])

    for i in range(dic_pred.shape[0]):
        acc[i] = dic_pred[i, i] / dic_pred[i, :].sum()
        print("第{:d}类正确率:{:.2f}".format(i, dic_pred[i, i] / dic_pred[i, :].sum()))
    print('总体正确率{:.4f}'.format(acc.mean()))
    if need_image == True:
        if not os.path.exists(str(num_classes)+type+'_Tsne_img'):
            os.makedirs(str(num_classes)+type+'_Tsne_img')
        heatmap(dic_pred,cmap='Blues',annot=True)
        plt.savefig(str(num_classes)+type+'_Tsne_img/' + 'heatmap_' + str(num_classes) + '_fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(
                patch_size) + 'acc' + str(acc.mean())[:5] + '.png')
        plt.show()
        data_t_sne = prepocessing_tsne(data, 2)
        # color = ['r','b','g','y','o','k']
        plt.scatter(data_t_sne[:, 0], data_t_sne[:, 1], c=label)
        plt.savefig(
            str(num_classes)+type+'_Tsne_img/' + 'skcls' + str(num_classes) + '_fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(
                patch_size) + 'acc' + str(acc.mean())[:5] + '.png')
        plt.show()
    else:
        return acc.mean()