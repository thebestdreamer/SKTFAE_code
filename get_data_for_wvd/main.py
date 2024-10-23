import numpy as np
from get_data_for_wvd.rml_data_class import RML_cls
from get_data_for_wvd.load_data import load_RML_data
from get_data_for_wvd.WVD import pwvd

def load_data(dataset,snr,cls_num = 4):
    data_np,real_label = load_RML_data(dataset = dataset, SNR = snr, norm='no')
    para = RML_cls(select_dataset = dataset)

    # different cls_num means different modulation class set
    if cls_num == 4:
        cls_list = ['AM-DSB','GFSK','PAM4','QAM64']
    if cls_num == 6:
        cls_list = ['PAM4', 'QPSK', 'QAM64', 'AM-DSB', 'GFSK','CPFSK']
    if cls_num == 7:
        cls_list = ['PAM4', 'QPSK', 'QAM64', 'AM-DSB', 'AM-SSB', 'GFSK', 'CPFSK']
    # cls_list = ['QAM16','PAM4','AM-SSB','8PSK','QPSK','QAM64','AM-DSB','WBFM','CPFSK','GFSK','BPSK']
    data = []
    label = []
    for pl,i in enumerate(cls_list):
        pl_data = para.cls_list.index(i)
        tem = data_np[0][pl_data==real_label[0]]
        for j in tem:
            data.append(pwvd(j))
        label.append(np.ones([tem.shape[0],1])*pl)
    label = np.concatenate(label, axis=0)
    data = np.concatenate(data,axis=0)
    return data,label.squeeze()

def main():
    snr = 18
    dataset = 'rml22'
    data,label = load_data(dataset,snr)
    # np.save('7class_'+dataset+'_img_data_snr'+str(snr)+'.npy',data)
    # np.save('save_dataset/7label',label)

if __name__ == '__main__':
    main()