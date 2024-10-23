from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def T_sne(fea,label):
    cls_num = 4
    cls_list = ['AM-DSB','GFSK','PAM4','QAM64']
    label = np.squeeze(label)
    # color = ['#ceff29', '#0000f1', '#7dff7a', '#010180','#29ffce','#00b1ff','#004dff','#5dffae','#ffc400','#f10800','#800000']
    color = ['#31688e','#440154','#35b779','#fde725']
    # color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#0780cf']
    tsne = TSNE(n_components=2).fit_transform(fea)
    # 创建一个新的figure
    plt.figure(0, figsize=(10, 10))
    cmap = 'jet'
    # 在第一个子图上绘制data_img1的TSNE结果
    # plt.scatter(tsne[:, 0], tsne[:,1], c=label, cmap=cmap)
    for i in range(cls_num):
        plt.scatter(tsne[label == i, 0], tsne[label == i, 1], c=color[i], cmap=cmap, label = cls_list[i] )
        plt.legend()
    plt.axis("equal")
    plt.title('TSNE of data_img1')
    plt.savefig('Tsne_img/' + 'cls' + str(cls_num) + 'fea_mask' + str(int(0.90 * 100)) + '_' + 'patch' + str(
        16) + '.png')
    plt.show()