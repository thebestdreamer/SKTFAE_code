from dataset_load_Mo_center import get_MAE_model_Mo
from dataset_load_trainable import get_MAE_model_train
from dataset_load_MAE import get_MAE_model_MAE
from fea_ext import get_fea_file
from kmeans import k_means
# from FCM import fcm
import numpy as np
import torch
from sk_cluster import sk_cluster

# 另外一组运行脚本， 方便同时运行多个模型，加快结果统计。
def main():
    # random seed
    seed = 3407
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 设定一系列基础参数
    alpha = 0.1
    beta = 20
    batch_size = 100
    warmup_epoch = 20
    epoch_sum = 300
    mask_ratio = 0.90
    base_learning_rate = 1e-4 #注意根据预训练和训练结果的差异调整学习率
    data_file = ['rml16a',6]
    # data_file = 'D:\\360安全浏览器下载\\资料文件\\0-待完成任务\\RML2016.10B\\vision\\4class_img_data_snr0.npy'
    patch_size = 16
    num_classes = 4
    #根据基础参数设置模型名称并进行 训练、特征提取、测试
    model_name = 'best_mask'+str(int(mask_ratio*100))+'_'+'patch'+str(patch_size)+'.pth'


    MAE_basic_train = False
    train_update_train = False

    ## MAE 掩码自监督
    if MAE_basic_train == True:
        # training the basic model for the first stage(clustering on origin feature)
        get_MAE_model_MAE(batch_size,warmup_epoch,epoch_sum,mask_ratio,patch_size,base_learning_rate,data_file,model_name,num_classes,alpha,beta)
        fea_name = str(num_classes) + 'fea_MAE_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
        label_name = str(num_classes)+'fea_MAE_label/'+'label.npy'
        get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,
                     str(num_classes)+'class_MAE_checkpoint/'+model_name,num_classes)

    ##Train 训练式更新方法
    if train_update_train == True:
        fea_name = str(num_classes) + 'fea_train_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
        label_name = str(num_classes) + 'fea_train_label/' + 'label.npy'
        get_fea_file(batch_size, mask_ratio, patch_size, data_file, fea_name,label_name,
                     str(num_classes)+'class_MAE_checkpoint/' + model_name, num_classes)
        get_MAE_model_train(batch_size, warmup_epoch, epoch_sum, mask_ratio, patch_size, base_learning_rate, data_file,
                     model_name, num_classes, alpha, beta)
        get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,
                    str(num_classes)+'class_train_checkpoint/'+model_name,num_classes)


    k_means(fea_name,label_name,num_classes,mask_ratio,patch_size)
    # fcm(fea_name,num_classes,mask_ratio,patch_size)
    sk_cluster(fea_name,num_classes,mask_ratio,patch_size,label_name,center_load=False,need_image=False)

if __name__ == '__main__':
    main()