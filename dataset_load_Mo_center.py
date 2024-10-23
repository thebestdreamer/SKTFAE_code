
from cuda_K_means import kmeans
import math
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from model import MAE_ViT
from Sinkhorn_cuda import compute_optimal_transport
from kmeans import kmeans_np
from scipy.optimize import linear_sum_assignment
import os
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


def match(center_old,center_new,gam):
    cost_matrix = torch.zeros([center_old.size()[0],center_new.size()[0]])
    for pl_i,i in enumerate(center_old):
        for pl_j,j in enumerate(center_new):
            cost_matrix[pl_i,pl_j] = ((i-j)**2).mean()

    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().numpy())

    center = gam * center_new[col_ind[:]] + (1 - gam) * center_old
    return center

def cross_entropy_fuzzy(input,target,t,index,save_epoch):
    one_hot_tensor = torch.zeros_like(input)
    for i in range(input.size()[0]):
        one_hot_tensor[i][target[i]] = 1
    weight = (torch.max(t,dim=1).values>index)
    Cen_loss = -1*one_hot_tensor*torch.log(input+1e-6)-(1-one_hot_tensor)*torch.log(1-input+1e-6)*0
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

def loss_fun(x,output,fea_mid,alpha=0.1,beta=0.1,num_cluster=4,save_epoch=0,center=None,center_old = None):
    L1 = torch.mean((x-output)**2)
    d = torch.zeros([x.size()[0],num_cluster]).to('cuda')
    if center_old is not None:
        gam = 0.1
        center_new = find_center_cuda(fea_mid[0], num_cluster)
        center = match(center_old, center_new, gam)
    for pl_i, i in enumerate(fea_mid[0]):
        for pl_j, j in enumerate(center):
            d[pl_i, pl_j] = torch.mean((i - j) ** 2)
    num, cls = d.shape[:]
    # selecting the nearest-neighbor-labeling or SK labeling
    with torch.no_grad():
        # NN Labeling
        # pseudo_label = torch.argmin(d, dim=1)
        # SK Labeling
        t, v = compute_optimal_transport(d, (torch.ones([num, ]) / num).to('cuda'),
                                         (torch.ones([cls, ]) / cls).to('cuda'), 20)
        for i, _ in enumerate(t):
            t[i, :] /= t[i, :].sum()
        pseudo_label = torch.argmax(t, dim=1)

    P = torch.exp(-beta*d)
    P = P / P.sum(dim=1, keepdim=True)
    # just cross entropy loss
    Cen_loss = cross_entropy_fuzzy(P, pseudo_label,P, 0,save_epoch)

    return L1 + alpha * Cen_loss, Cen_loss, center

def get_MAE_model_Mo(batch_size,warmup_epoch,epoch_sum,mask_ratio,patch_size,base_learning_rate,data_file,model_name,num_cls,alpha,beta):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split_ratio = 0.3
    data,label = load_data(data_file[0],data_file[1],num_cls)
    allset = dataset(data,label)
    center_name = 'center' + '_fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    train_data,test_data = torch.utils.data.random_split(allset,[int(len(allset)*(1-split_ratio)),len(allset)-int(len(allset)*(1-split_ratio))])
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=0,drop_last=True
                              ,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,num_workers=0)
    # selecting the setting of MAEVIT
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
    model.load_state_dict(torch.load(str(num_cls)+'class_MAE_checkpoint/' + model_name))
    train(model,train_loader,test_loader,batch_size,patch_size,base_learning_rate,mask_ratio,warmup_epoch,epoch_sum,data_file,model_name,num_cls,alpha,beta,center_name)


def train(model,train_loader,test_loader,batch_size,patch_size,base_learning_rate,mask_ratio,
          warmup_epoch,epoch_sum,data_file,model_name='tem.pth',num_cls = 11,alpha=0.1,beta=0.1,center_name = 'center.npy'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    min_loss = 10000
    flag = 0
    save_epoch = 0
    center_old = None
    fea_name = str(num_cls) + 'fea_Mo_label/' + 'fea_mask' + str(int(mask_ratio * 100)) + '_' + 'patch' + str(patch_size) + '.npy'
    fea_data = np.load(fea_name)
    center = torch.from_numpy(kmeans_np(fea_data,num_cls,need_center=True)).to('cuda')
    optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=base_learning_rate * batch_size / 256,betas=(0.9,0.99),
                                  weight_decay=1e-6)
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / epoch_sum * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    for epoch in range(epoch_sum):
        loss_all = []
        fea_list = []
        label_list = []
        with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_sum}', postfix=dict, mininterval=0.3) as pbar:
            for data, label in train_loader:
                data = data.float()
                data = data.to(device)
                optimizer.zero_grad()
                output, mask, fea_mid = model(data)
                fea_list.append(fea_mid[0].detach().to('cpu').numpy())
                label_list.append(label.detach().numpy())
                loss,cen_loss,center_old = loss_fun(data, output, fea_mid,alpha,beta,num_cls,save_epoch=save_epoch,center=center,center_old=center_old)
                loss.backward()
                optimizer.step()
                if cen_loss != 0:
                    loss_all.append(loss.cpu().detach().numpy())
                    train_loss = np.array(loss_all).mean()
                #为便于观测损失变化，将平均损失增大100倍
                pbar.set_postfix(**{'train_loss': loss.cpu().detach().numpy(),"mean_loss:":100*train_loss.item()})
                pbar.update(1)
        #early_stopping
        #需要考虑center是否参与训练，从而输入test不同的center
        new_loss = test(model,test_loader,mask_ratio,save_epoch,alpha,beta,num_cls,center_old)
        lr_scheduler.step()
        if new_loss<=min_loss:
            print("Update")
            save_epoch = 0#epoch
            min_loss = new_loss
            detect(str(num_cls)+'class_Mo_checkpoint')
            detect(str(num_cls)+'fea_Mo_label')
            torch.save(model.state_dict(),str(num_cls)+'class_Mo_checkpoint/'+model_name)
            np.save(str(num_cls)+'fea_Mo_label/'+center_name,center_old.to('cpu').detach().numpy())
            flag = 0
        else:
            flag += 1
            print("Early stop{:d}/20".format(flag))

        fea_np = np.concatenate(fea_list,axis=0)
        label_np = np.concatenate(label_list,axis=0)
        if flag >= 20:
            print('Early Stopping')
            break

def test(model,test_loader,mask_ratio,save_epoch,alpha,beta,num_cls,center):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    acc = []
    loss_all = []
    with tqdm(total=len(test_loader), postfix=dict, mininterval=0.3) as pbar:
        for data, label in test_loader:
            data = data.float()
            label = label.long()
            data = data.to(device)
            output,mask,fea_mid = model(data)
            loss,cen_loss,_ = loss_fun(data, output, fea_mid,alpha,beta,num_cls,save_epoch=save_epoch,center=center)
            if cen_loss != 0:
                loss_all.append(loss.cpu().detach().numpy())
            pbar.set_postfix(**{'loss': loss.item()})
            pbar.update(1)
        pbar.set_description("mean_loss: %.4f" %(np.array(loss_all).mean()*100))
    return np.array(loss_all).mean()

if __name__ == '__main__':
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()
