from tftb.generators import anapulse
from tftb.processing import WignerVilleDistribution
from tftb.processing import PseudoWignerVilleDistribution
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt

def stp(data,nperseg=10,noverlap=9):
    fs = 200*1000
    sgn=data[0,:]+data[1,:]*1j
    f, tt, p = spectrogram(x=sgn, fs=fs, nperseg=nperseg,
                           noverlap=noverlap,
                           return_onesided=False)
    # ampLog =np.abs(np.log(1 + np.abs(p)))
    #ampLog=np.abs(p)+np.random.normal(0,size=(51,290))
    normImg =p/np.linalg.norm(p)
    normImg =np.expand_dims(normImg, axis=0)
    # threshold = 0.04
    # normImg[np.where(normImg > threshold)] = threshold
    # plt.pcolormesh(normImg)
    normImg =  (p - p.min()) / (p.max() - p.min() )  # 归一化
    return tt,f,normImg



def pwvd(data):

    t = np.linspace(0, 1, data.shape[1])
    sgn = data[0, :] + data[1, :] * 1j
    spec = PseudoWignerVilleDistribution(sgn, timestamps=t, n_fbins=128)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)
    img,_,_=spec.run()
    # ampLog = img
    # img = (ampLog - ampLog.min()) / (ampLog.max() - ampLog.min())
    img=(img - np.mean(img)) / (np.var(img) + 1.e-6) ** .5
    img=np.expand_dims(img, axis=0)

    return img

def mask(img,patch_size,mask_ratio):
    mask = np.linspace(0,1,img.shape[0]*img.shape[1]//patch_size**2)
    np.random.shuffle(mask)
    mat = np.reshape(mask,(np.array(img.shape)//patch_size))
    mat = mat>mask_ratio
    mask_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mask_img[i,j] = img[i,j] * mat[i//patch_size,j//patch_size]
    return mask_img


def main():
    for snr in range(18, 20, 2):
        snr = int(snr)
        cls = ['AM-DSB', 'GFSK', 'PAM4', 'QAM64']
        dic_mat = np.load(
            'D:\\360安全浏览器下载\\资料文件\\0-待完成任务\\RML2016.10a\\save_dataset\\4class_data_snr' + str(snr) + '.npy')
        save_img = []
        pl = 0
        for i in range(0, dic_mat.shape[0], 1000):
            data = dic_mat[i, :, :]
            plt.figure()
            plt.plot(data[0], linewidth=3)
            plt.plot(data[1], linewidth=3, c=(1, 182 / 255, 124 / 255))
            # plt.show()
            plt.savefig('lin_' + cls[pl] + '.jpg', format='jpg')

            img = pwvd(data)
            # tt,f,img = stp(data)
            # plt.figure(figsize=(6, 5))
            # plt.pcolormesh(tt, f, img, shading='goudard')
            # plt.show()
            # plt.savefig('tem_' + cls[pl] + '.jpg')
            # pl += 1
            plt.figure()
            save_img.append(img)
            plt.pcolormesh(img[0])
            # plt.show()
            plt.savefig('tem_' + cls[pl] + '.jpg')
            pl += 1
            # mask_img = mask(img,16,0.9)
            # plt.pcolormesh(mask_img)
            # plt.show()
        # save_img = np.array(save_img)
        # np.save('4class_img_data_snr' + str(snr) + '.npy', save_img)
        # img = pwvd(data)
        # img = img[0,:,:]

    # x= np.linspace(0,127,128)
    # plt.plot(x,data[0],c=(0.8,0.2,0.2),linewidth=3)

if __name__ == '__main__':
    main()