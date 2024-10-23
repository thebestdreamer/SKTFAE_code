import sklearn.cluster as cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from T_sne import T_sne
#
def prepocessing_tsne(data, n):
    starttime_tsne = time.time()
    dataset = TSNE(n_components=n, random_state=33).fit_transform(data)
    endtime_tsne = time.time()
    print('cost time by tsne:', endtime_tsne - starttime_tsne)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tsne = scaler.fit_transform(dataset)
    return X_tsne


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

def swap(x):
    for i in range(x.shape[0]):
        t = np.argmax(x[i,:])
        if x[i,t]>x[t,t]:
            tem = x[:,i].copy()
            x[:,i] = x[:,t]
            x[:,t] = tem
    return x


def randperm(n):
    # 生成一个包含0到n-1的整数数组
    arr = np.arange(n)
    # 随机打乱数组元素的顺序
    np.random.shuffle(arr)
    return arr

def kmeans_np(data, k, max_time=100,need_center = False):
    # the K-means base on numpy array
    n, m = data.shape
    ini = randperm(n)[:k]  # 只有一维需要逗号
    midpoint = data[ini]  # 随机选择k个起始点
    time = 0
    last_label = 0
    while (time < max_time):
        d = np.repeat(data[np.newaxis,:,:],k,axis=0)  # shape k*n*m
        mid_ = np.repeat(midpoint[:,np.newaxis,:],n,axis=1)  # shape k*n*m
        dis = -np.sum(d * mid_, axis=2)  # 计算距离
        label = dis.argmin(0)  # 依据最近距离标记label
        if np.sum(label != last_label) == 0:  # label没有变化,跳出循环
            if need_center:
                return midpoint
            else:
                return label
        last_label = label
        for i in range(k):  # 更新类别中心点，作为下轮迭代起始
            kpoint = data[label == i]
            if time == 0:
                if i == 0:
                    midpoint = kpoint.mean(0)[np.newaxis,:]
                else:
                    midpoint = np.concatenate([midpoint, kpoint.mean(0)[np.newaxis,:]], 0)
            else:
                midpoint[i] = kpoint.mean(0)[np.newaxis,:]
        time += 1
    if need_center:
        return midpoint
    else:
        return label

def main():
    np.random.seed(42)

    data = np.load('fea_label/fea.npy')
    label = np.load('fea_label/label.npy')

    print(data.shape)

    num_classes = 4
    # 归一化 0.47
    # for i in range(0,data.shape[0]):
    #     data[i,:] = (np.e**data[i,:])/(np.e**data[i,:]).sum()

    # argmax 0.25
    # one_hot = np.zeros_like(data)
    # for i in range(0,data.shape[0]):
    #     one_hot[i, np.argmax(data[i,:])] = 1
    # data = one_hot

    # 正则化 0.46
    # mean = data.mean(0)
    # std = data.std(0)
    # for i in range(0,data.shape[0]):
    #     data[i,:] = (data[i,:]-mean)/std

    pca = PCA(2)
    data_pca = pca.fit_transform(data[:, :])
    clu = cluster.KMeans(n_clusters=num_classes, random_state=42)
    # clu = cluster.DBSCAN(eps=3,min_samples=80,random_state=42)

    y_pred = clu.fit_predict(data[:, :])

    y_ls = list(y_pred)
    dic_pred = np.zeros([num_classes, num_classes])
    for i in range(len(y_ls)):
        dic_pred[np.int(label[i]), y_pred[i]] += 1

    # sklearn自带算法  DBI的值最小是0，值越小，代表聚类效果越好。
    cluster_score_DBI = metrics.davies_bouldin_score(data, y_pred)
    cluster_score_real_DBI = metrics.davies_bouldin_score(data, label)
    cluster_score_NMI = metrics.normalized_mutual_info_score(label, y_pred)
    cluster_score_F = metrics.f1_score(label, y_pred, average='micro')
    print("cluster_score_DBI:", cluster_score_DBI)
    print('cluster_score_real_DBI', cluster_score_real_DBI)
    print("cluster_score_NMI:", cluster_score_NMI)
    print('cluster_score_F:', cluster_score_F)

    # print(dic_pred)
    dic_pred = swap(dic_pred)
    print(dic_pred)
    acc = np.zeros([dic_pred.shape[0], 1])

    for i in range(dic_pred.shape[0]):
        acc[i] = dic_pred[i, i] / dic_pred[i, :].sum()
        print("第{:d}类正确率:{:.2f}".format(i, dic_pred[i, i] / dic_pred[i, :].sum()))
    print('总体正确率{:.4f}'.format(acc.mean()))

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=label[:])
    plt.show()


def find_center(x,mask_ratio,patch_size,num_cluster):
    clu = cluster.KMeans(n_clusters=num_cluster,random_state=42)
    tem = x
    y_pred = clu.fit_predict(tem)
    center_list = []
    for i in range(num_cluster):
        center_list.append(tem[np.argwhere(y_pred==i)].mean(axis=0))
    return np.array(center_list)

def k_means(fea_name,label_name, num_classes=4,mask_ratio=0.75,patch_size=16):
    np.random.seed(3407)
    data = np.load(fea_name)
    label = np.load(label_name)
    print(data.shape)

    # 归一化 0.47
    # for i in range(0,data.shape[0]):
    #     data[i,:] = (np.e**data[i,:])/(np.e**data[i,:]).sum()

    # argmax 0.25
    # one_hot = np.zeros_like(data)
    # for i in range(0,data.shape[0]):
    #     one_hot[i, np.argmax(data[i,:])] = 1
    # data = one_hot

    # 正则化 0.46
    # mean = data.mean(0)
    # std = data.std(0)
    # for i in range(0,data.shape[0]):
    #     data[i,:] = (data[i,:]-mean)/std

    pca = PCA(2)
    data_pca = pca.fit_transform(data[:, :])
    clu = cluster.KMeans(n_clusters=num_classes, random_state=42)
    # clu = cluster.DBSCAN(eps=3,min_samples=80,random_state=42)
    # find_center(data,mask_ratio,patch_size,num_classes)
    y_pred = clu.fit_predict(data[:, :])

    y_ls = list(y_pred)
    dic_pred = np.zeros([num_classes, num_classes])
    for i in range(len(y_ls)):
        dic_pred[np.int32(label[i]), y_pred[i]] += 1

    # sklearn自带算法  DBI的值最小是0，值越小，代表聚类效果越好。
    cluster_score_DBI = metrics.davies_bouldin_score(data, y_pred)
    cluster_score_real_DBI = metrics.davies_bouldin_score(data, label)
    cluster_score_NMI = metrics.normalized_mutual_info_score(label, y_pred)
    cluster_score_F = metrics.f1_score(label, y_pred, average='micro')
    print("cluster_score_DBI:", cluster_score_DBI)
    print('cluster_score_real_DBI', cluster_score_real_DBI)
    print("cluster_score_NMI:", cluster_score_NMI)
    print('cluster_score_F:', cluster_score_F)

    # print(dic_pred)
    dic_pred = swap(dic_pred)
    print(dic_pred)
    acc = np.zeros([dic_pred.shape[0], 1])

    for i in range(dic_pred.shape[0]):
        acc[i] = dic_pred[i, i] / dic_pred[i, :].sum()
        print("第{:d}类正确率:{:.2f}".format(i, dic_pred[i, i] / dic_pred[i, :].sum()))
    print('总体正确率{:.4f}'.format(acc.mean()))

    # plt.scatter(data_pca[:, 0], data_pca[:, 1], c=label)
    # plt.show()

    # data_t_sne = prepocessing_tsne(data, 2)
    # plt.scatter(data_t_sne[:, 0], data_t_sne[:, 1], c=label)
    # T_sne(data,label)
    return acc.mean()
if __name__ == '__main__':
    main()