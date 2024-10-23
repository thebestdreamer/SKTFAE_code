from dataset_load_Mo_center import get_MAE_model_Mo
from dataset_load_trainable import get_MAE_model_train
from dataset_load_MAE import get_MAE_model_MAE
from fea_ext import get_fea_file
from kmeans import k_means
# from FCM import fcm
import numpy as np
import torch
from sk_cluster import sk_cluster

# 在确定合适的超参数后，采用不同的随机数种子，统计对应的结果。
def Train(seed,SNR):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 设定一系列基础参数
    alpha = 0.1
    beta = 20
    split_ratio = 0.3
    batch_size = 150
    warmup_epoch = 20
    epoch_sum = 300
    mask_ratio = 0.90
    base_learning_rate = 1e-4 #注意根据预训练和训练结果的差异调整学习率
    data_file = ['rml16a',SNR]
    # data_file = 'D:\\360安全浏览器下载\\资料文件\\0-待完成任务\\RML2016.10B\\vision\\4class_img_data_snr0.npy'
    patch_size = 16
    num_classes = 4
    #根据基础参数设置模型名称并进行 训练、特征提取、测试
    model_name = 'best_mask'+str(int(mask_ratio*100))+'_'+'patch'+str(patch_size)+'.pth'

    ##Train 训练式更新方法
    fea_name = str(num_classes) + 'fea_train_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    label_name = str(num_classes) + 'fea_train_label/' + 'label.npy'
    get_fea_file(batch_size, mask_ratio, patch_size, data_file, fea_name,label_name,
                 str(num_classes)+'class_MAE_checkpoint/' + model_name, num_classes)
    get_MAE_model_train(batch_size, warmup_epoch, epoch_sum, mask_ratio, patch_size, base_learning_rate, data_file,
                 model_name, num_classes, alpha, beta)
    get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,
                str(num_classes)+'class_train_checkpoint/'+model_name,num_classes)

    ## MAE 掩码自监督
    # get_MAE_model_MAE(batch_size,warmup_epoch,epoch_sum,mask_ratio,patch_size,base_learning_rate,data_file,model_name,num_classes,alpha,beta)
    # fea_name = str(num_classes) + 'fea_MAE_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    # label_name = str(num_classes)+'fea_MAE_label/'+'label.npy'
    # get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,
    #              str(num_classes)+'class_MAE_checkpoint/'+model_name,num_classes)


    acc = k_means(fea_name,label_name,num_classes,mask_ratio,patch_size)
    # fcm(fea_name,num_classes,mask_ratio,patch_size)
    sk_acc = sk_cluster(fea_name,num_classes,mask_ratio,patch_size,label_name,center_load=False,need_image=False)
    return acc, sk_acc

def Mo(seed,SNR):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 设定一系列基础参数
    alpha = 0.01
    beta = 20
    split_ratio = 0.3
    batch_size = 50
    warmup_epoch = 20
    epoch_sum = 300
    mask_ratio = 0.90
    base_learning_rate = 1e-4 #注意根据预训练和训练结果的差异调整学习率
    data_file = ['rml16a',SNR]
    patch_size = 16
    num_classes = 4
    #根据基础参数设置模型名称并进行 训练、特征提取、测试
    model_name = 'best_mask'+str(int(mask_ratio*100))+'_'+'patch'+str(patch_size)+'.pth'

    ##Train 训练式更新方法
    fea_name = str(num_classes) + 'fea_Mo_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    label_name = str(num_classes) + 'fea_Mo_label/' + 'label.npy'
    get_fea_file(batch_size, mask_ratio, patch_size, data_file, fea_name,label_name,
                 str(num_classes)+'class_MAE_checkpoint/' + model_name, num_classes)
    get_MAE_model_Mo(batch_size, warmup_epoch, epoch_sum, mask_ratio, patch_size, base_learning_rate, data_file,
                model_name, num_classes, alpha, beta)

    get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,
               str(num_classes)+'class_Mo_checkpoint/'+model_name,num_classes)

    ## MAE 掩码自监督
    # get_MAE_model_MAE(batch_size,warmup_epoch,epoch_sum,mask_ratio,patch_size,base_learning_rate,data_file,model_name,num_classes,alpha,beta)
    # fea_name = str(num_classes) + 'fea_MAE_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    # label_name = str(num_classes)+'fea_MAE_label/'+'label.npy'
    # get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,
    #              str(num_classes)+'class_MAE_checkpoint/'+model_name,num_classes)


    acc = k_means(fea_name,label_name,num_classes,mask_ratio,patch_size)
    # fcm(fea_name,num_classes,mask_ratio,patch_size)
    sk_acc = sk_cluster(fea_name,num_classes,mask_ratio,patch_size,label_name,center_load=False,need_image=False)
    return acc, sk_acc

def write_acc_log(SNR,M_acc,T_acc,M_sk_acc,T_sk_acc):
    with open('acc_log.txt', 'a+') as  f:
        dic = {}
        dic['SNR'] = SNR
        dic['Mo_acc'] = M_acc
        dic['Train_acc'] = T_acc
        dic['Mo_sk_acc'] = M_sk_acc
        dic['Train_sk_acc'] = T_sk_acc
        f.write(str(dic))
        f.write('\n')

def main():
    seed_list = [42,3407,2024,1437,1234]
    SNR_list = [2,4,6,8,10,12,14,16,18]
    for SNR in SNR_list:
        T_acc_list = []
        T_sk_acc_list = []
        M_acc_list = []
        M_sk_acc_list = []
        for seed in seed_list:
            T_acc,T_sk_acc = Train(seed,SNR)
            M_acc,M_sk_acc = Mo(seed,SNR)
            T_acc_list.append(T_acc)
            M_acc_list.append(M_acc)
            T_sk_acc_list.append(T_sk_acc)
            M_sk_acc_list.append(M_sk_acc)
            write_acc_log(SNR, M_acc, T_acc, M_sk_acc, T_sk_acc)
        T_mean_acc = (sum(T_acc_list) / len(T_acc_list))
        M_mean_acc = (sum(M_acc_list) / len(M_acc_list))
        T_sk_mean_acc = (sum(T_sk_acc_list) / len(T_sk_acc_list))
        M_sk_mean_acc = (sum(M_sk_acc_list) / len(M_sk_acc_list))
        write_acc_log(SNR,M_mean_acc,T_mean_acc,T_sk_mean_acc,M_sk_mean_acc)

if __name__ == '__main__':
    main()