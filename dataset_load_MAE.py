import os.path
import math
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from model import MAE_ViT
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

def detect(path):
    if not os.path.exists(path):
        os.makedirs(path)



def get_MAE_model_MAE(batch_size,warmup_epoch,epoch_sum,mask_ratio,patch_size,base_learning_rate,data_file,model_name,num_cls,alpha,beta):
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
    model.load_state_dict(torch.load(str(num_cls)+'class_MAE_checkpoint/' + model_name))
    # model.parameters()
    train(model,train_loader,test_loader,batch_size,patch_size,base_learning_rate,mask_ratio,warmup_epoch,epoch_sum,data_file,model_name,num_cls,alpha,beta,center_name)


def train(model,train_loader,test_loader,batch_size,patch_size,base_learning_rate,mask_ratio,
          warmup_epoch,epoch_sum,data_file,model_name='tem.pth',num_cls = 11,alpha=0.1,beta=0.1,center_name = 'center.npy'):
    loss_mean = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    min_loss = 10000
    flag = 0
    optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=base_learning_rate * batch_size / 256,betas=(0.9,0.95),
                                  weight_decay=0.01)
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / epoch_sum * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    for epoch in range(epoch_sum):
        loss_all = []
        with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_sum}', postfix=dict, mininterval=0.3) as pbar:
            for data, label in train_loader:
                # data = torch.from_numpy(data)
                # label = torch.from_numpy(label)
                data = data.float()
                data = data.to(device)
                # label = label.long()
                # label = label.to(device)
                optimizer.zero_grad()
                output, mask, _ = model(data)
                loss = torch.mean((data - output)**2) #** 2 * mask) / mask_ratio
                loss.backward()
                optimizer.step()
                loss_all.append(loss.cpu().detach().numpy())
                train_loss = np.array(loss_all).mean()
                # pbar.set_description("mean_loss: %.4f" % train_loss.item())
                pbar.set_postfix(**{'train_loss_': loss.cpu().detach().numpy(),"mean_loss:":train_loss.item()})
                pbar.update(1)
        #early_stopping
        #需要考虑center是否参与训练，从而输入test不同的center
        new_loss = test(model,test_loader)
        loss_mean.append(new_loss)
        lr_scheduler.step()
        if new_loss<=min_loss:
            print("Update")
            min_loss = new_loss
            detect(str(num_cls)+'class_MAE_checkpoint')
            detect(str(num_cls)+'fea_MAE_label')
            torch.save(model.state_dict(),str(num_cls)+'class_MAE_checkpoint/'+model_name)
            flag = 0
        else:
            flag += 1
        if flag >= 40:
            plt.plot(loss_mean * 100)
            plt.show()
            print('Early Stopping')
            break

def test(model,test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    loss_all = []
    with tqdm(total=len(test_loader), postfix=dict, mininterval=0.3) as pbar:
        for data, label in test_loader:
            data = data.float()
            label = label.long()
            data = data.to(device)
            label = label.to(device)
            output,mask,_ = model(data)
            loss = torch.mean((data - output)**2)
            pbar.set_postfix(**{'loss': loss.item()})
            pbar.update(1)
            loss_all.append(loss.cpu().detach().numpy())
        pbar.set_description("mean_loss: %.4f" %(np.array(loss_all).mean()*100))
    return np.array(loss_all).mean()

if __name__ == '__main__':
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()
