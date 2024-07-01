
from cuda_K_means import kmeans
import math
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from model import MAE_ViT
import os
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.nn.functional import cross_entropy
from Sinkhorn_cuda import compute_optimal_transport
from kmeans import kmeans_np
# from entropy.dse import diffusion_spectral_entropy
# from entropy.dsmi import diffusion_spectral_mutual_information
from fea_ext import get_fea_file
from sk_cluster import sk_cluster
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

def detect(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cross_entropy_fuzzy(input,target,t,index):
    one_hot_tensor = torch.zeros_like(input)
    for i in range(input.size()[0]):
        one_hot_tensor[i][target[i]] = 1
    weight = (torch.max(input,dim=1).values>=index)
    Cen_loss = -one_hot_tensor*torch.log(input+1e-6)#-(1-one_hot_tensor)*torch.log(1-input+1e-6)
    return (weight*Cen_loss.mean(1)).sum()/(weight.sum()+1e-3)

def find_center_cuda(x,num_cluster):
    tem = x.detach()
    y_pred = kmeans(tem,num_cluster)
    center_list = torch.zeros([num_cluster,x.size()[1]]).to('cuda')
    for i in range(num_cluster):
        if (y_pred==i).sum() == 0:
            pass
        else:
            center_list[i]=tem[torch.where(y_pred==i)].mean(axis=0)
    return center_list

def loss_fun(x,output,fea_mid,alpha=0.1,beta=0.1,num_cluster=4,epoch=0,center=None):
    L1 = torch.mean((x-output)**2)
    d = torch.zeros([x.size()[0],num_cluster]).to('cuda')
    for pl_i, i in enumerate(fea_mid[0]):
        for pl_j, j in enumerate(center):
            d[pl_i, pl_j] = torch.mean((i - j) ** 2)
    num, cls = d.shape[:]
    # for i, _ in enumerate(d):
    #     d[i, :] /= d[i, :].sum()
    pseudo_label = torch.argmin(d, dim=1)
    # with torch.no_grad():
        # t, v = compute_optimal_transport(d, (torch.ones([num, ]) / num).to('cuda'),
        #                                  (torch.ones([cls, ]) / cls).to('cuda'), 20)
        # for i, _ in enumerate(t):
        #     t[i, :] /= t[i, :].sum()
        # pseudo_label = torch.argmax(t, dim=1)
    #
    # P = torch.exp(-beta*d)
    # P = P / P.sum(dim=1,keepdim=True)


    P = torch.exp(-beta*d)
    pos = P / P.sum(dim=1,keepdim=True)
    t = 0
    Cen_loss = cross_entropy_fuzzy(pos, pseudo_label,t,0)
    # Cen_loss = torch.nn.functional.cross_entropy(pos,pseudo_label)
    # 概率*距离并给定伪标签构造损失，指导网络向目标类别最小距离收敛
    # Cen_loss = (pos[:,pseudo_label]*(P[:,pseudo_label])).mean()

    return L1 + alpha * Cen_loss, Cen_loss


def get_MAE_model_train(batch_size,warmup_epoch,epoch_sum,mask_ratio,patch_size,base_learning_rate,data_file,model_name,num_cls,alpha,beta):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split_ratio = 0.3
    data,label = load_data(data_file[0],data_file[1],num_cls)
    allset = dataset(data,label)
    center_name = 'center' + '_fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    train_data,test_data = torch.utils.data.random_split(allset,[int(len(allset)*(1-split_ratio)),len(allset)-int(len(allset)*(1-split_ratio))])
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=0,drop_last=True
                              ,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,num_workers=0)

    model = MAE_ViT(image_size=128,
                    patch_size=patch_size,
                    emb_dim=192,
                    mask_ratio=mask_ratio
                    )
    # model = MAE_ViT(image_size=128,
    #                 patch_size=patch_size,
    #                 emb_dim=64,
    #                 mask_ratio=mask_ratio,
    #                 encoder_layer=4,
    #                 decoder_layer=4,
    #                 encoder_head=8,
    #                 decoder_head=8
    #                 )
    model.load_state_dict(torch.load(str(num_cls)+'class_MAE_checkpoint/' + model_name))
    # model.parameters()
    train(model,train_loader,test_loader,batch_size,patch_size,base_learning_rate,mask_ratio,warmup_epoch,epoch_sum,data_file,model_name,num_cls,alpha,beta,center_name)


def train(model,train_loader,test_loader,batch_size,patch_size,base_learning_rate,mask_ratio,
          warmup_epoch,epoch_sum,data_file,model_name='tem.pth',num_cls = 11,alpha=0.1,beta=0.1,center_name = 'center.npy'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    min_loss = 10000
    flag = 0
    fea_name = str(num_cls) + 'fea_train_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    fea_data = np.load(fea_name)
    center = Parameter(torch.from_numpy(kmeans_np(fea_data,num_cls,need_center=True)).to('cuda'))
    center.requires_grad = True
    optimizer = torch.optim.AdamW([{"params": model.parameters()},
                       {"params": center}], lr=base_learning_rate * batch_size / 256,betas=(0.9,0.99),
                                  weight_decay=0.01)
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / epoch_sum * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    for epoch in range(epoch_sum):
        loss_all = []
        fea_list = []
        label_list = []
        with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_sum}', postfix=dict, mininterval=0.3) as pbar:
            for data, label in train_loader:
                # data = torch.from_numpy(data)
                # label = torch.from_numpy(label)
                data = data.float()
                data = data.to(device)
                # label = label.long()
                # label = label.to(device)
                optimizer.zero_grad()
                output, mask, fea_mid = model(data)
                fea_list.append(fea_mid[0].detach().to('cpu').numpy())
                label_list.append(label.detach().numpy())
                #loss = torch.mean((data - output)**2) #** 2 * mask) / mask_ratio
                loss,cen_loss = loss_fun(data, output, fea_mid,alpha,beta,num_cls,epoch=epoch,center=center)
                loss.backward()
                optimizer.step()
                if cen_loss != 0:
                    loss_all.append(loss.cpu().detach().numpy())
                    train_loss = np.array(loss_all).mean()
                # pbar.set_description("mean_loss: %.4f" % train_loss.item())
                #为便于观测损失变化，将平均损失增大100倍
                pbar.set_postfix(**{'train_loss_': loss.cpu().detach().numpy(),"mean_loss:":100*train_loss.item()})
                pbar.update(1)
        #early_stopping
        #需要考虑center是否参与训练，从而输入test不同的center
        new_loss = test(model,test_loader,mask_ratio,epoch,alpha,beta,num_cls,center)
        lr_scheduler.step()
        if new_loss<=min_loss:
            print("Update")
            min_loss = new_loss
            detect(str(num_cls)+'class_Train_checkpoint')
            detect(str(num_cls)+'fea_Train_label')
            torch.save(model.state_dict(),str(num_cls)+'class_Train_checkpoint/'+model_name)
            np.save(str(num_cls)+'fea_Train_label/'+center_name,center.to('cpu').detach().numpy())
            flag = 0
        else:
            flag += 1
            print("Early stop{:d}/20".format(flag))

        fea_np = np.concatenate(fea_list,axis=0)
        label_np = np.concatenate(label_list,axis=0)
        if flag >= 20:
            print('Early Stopping')
            break

def test(model,test_loader,mask_ratio,epoch,alpha,beta,num_cls,center):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    acc = []
    loss_all = []
    with tqdm(total=len(test_loader), postfix=dict, mininterval=0.3) as pbar:
        for data, label in test_loader:
            # data = torch.from_numpy(data)
            # label = torch.from_numpy(label)
            data = data.float()
            label = label.long()
            data = data.to(device)
            label = label.to(device)
            output,mask,fea_mid = model(data)
            loss,cen_loss = loss_fun(data, output, fea_mid,alpha,beta,num_cls,epoch=epoch,center=center)
            if cen_loss != 0:
                loss_all.append(loss.cpu().detach().numpy())
            # now_acc = torch.sum(output.argmax(axis=1)==label)/len(label)
            # acc.append(now_acc.cpu().detach().numpy())
            # pbar.set_description("loss: %.4f" % loss.item())
            pbar.set_postfix(**{'loss': loss.item()})
            pbar.update(1)
        pbar.set_description("mean_loss: %.4f" %(np.array(loss_all).mean()*100))
    return np.array(loss_all).mean()

if __name__ == '__main__':
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()
