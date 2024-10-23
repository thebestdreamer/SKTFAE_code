#本文件用于从best.pth模型中提取给定数据的深层特征，以供后续聚类过程的应用

import os
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import math
from einops import repeat, rearrange
from tqdm import tqdm
from model import MAE_ViT
# from entropy.dse import diffusion_spectral_entropy
# from entropy.dsmi import diffusion_spectral_mutual_information
import matplotlib.pyplot as plt
from get_data_for_wvd.main import load_data
class dataset(Dataset):
    def __init__(self,data,label):
        self.label = label
        self.data = data[:,np.newaxis]
        # self.label = label
    def __getitem__(self, item):
        data_item = self.data[item,:,:]
        # label = self.label[item]
        # data = np.load('image/'+str(item)+'.npy')
        label = self.label[item]
        return data_item,label
    def __len__(self):
        return self.data.shape[0]

def get_fea_file(batch_size,mask_ratio,patch_size,data_file,fea_name,label_name,model_name,num_cls):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data,label = load_data(data_file[0],data_file[1],num_cls)
    allset = dataset(data,label)
    test_loader = DataLoader(allset,batch_size=batch_size,num_workers=0,shuffle=True)
    model = MAE_ViT(image_size=128,
                    patch_size=patch_size,
                    emb_dim=192,
                    mask_ratio=mask_ratio
                    )
    # model = MAE_ViT(image_size=128,
    #                 patch_size=patch_size,
    #                 emb_dim=192,
    #                 mask_ratio=mask_ratio,
    #                 encoder_layer=4,
    #                 decoder_layer=4,
    #                 encoder_head=6,
    #                 decoder_head=6
    #                 )
    model.load_state_dict(torch.load(model_name))
    label_ls, fea_ls = test(model, test_loader, mask_ratio)
    label_ls = np.concatenate(label_ls, axis=0)
    fea_ls = np.concatenate(fea_ls, axis=0)
    np.save(label_name,label_ls)
    np.save(fea_name,fea_ls)


def test(model,test_loader,mask_ratio):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    label_ls = []
    fea_ls = []
    dse_ls = []
    dsmi_ls = []
    with tqdm(total=len(test_loader), postfix=dict, mininterval=0.3) as pbar:
        for data, label in test_loader:
            # data = torch.from_numpy(data)
            # label = torch.from_numpy(label)
            data = data.float()
            data = data.to(device)
            output, mask, feature = model(data)
            label_ls.append(label)
            fea_ls.append(feature[0,:,:].detach().cpu().numpy())
            feature = rearrange(feature,'p b d->b (p d)')
            # dse_ls.append(diffusion_spectral_entropy(feature.detach().cpu().numpy(), gaussian_kernel_sigma=10, t=1))
            # dsmi_ls.append(diffusion_spectral_mutual_information(feature.detach().cpu().numpy(), label.detach().numpy(),n_clusters=len(set(label.numpy())))[0])
            pbar.update(1)
    # print('中间熵:',sum(dse_ls)/len(dse_ls))
    # print('中间标签互信息:', sum(dsmi_ls) / len(dsmi_ls))
    return label_ls, fea_ls

if __name__ == '__main__':
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()

