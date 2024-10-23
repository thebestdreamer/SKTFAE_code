import os
import pickle
import numpy as np
import torch

# from utils.norm import normalize_sig
def save_pkl(array,filename):
	with open(filename, 'wb') as f:
		pickle.dump(array, f)

def get_txt(file_name):
	data = np.loadtxt(file_name,complex)
	data_new = np.zeros([6000,2,128])
	data_new[:,0,:] = data.real
	data_new[:,1,:] = data.imag
	return data_new

def get_npy(file_name):
	data = np.load(file_name)
	data = np.swapaxes(data,1,2)
	return data
def collect_16b():
	cls_list = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	snr_list = np.linspace(-20,18,20).astype(np.int32)
	# snr_list = [18]
	data_dic = {}
	for pl,i in enumerate(cls_list):
		for snr in snr_list:
			key = (i,snr)
			data_dic[key] = get_txt(os.path.join(os.getcwd(),r'dataset/RML2016b_txt/'+i+' '+str(snr)+'.txt'))
	return data_dic

def collect_18a():
	cls_list = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK','32APSK',
			   '64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC',
			   'AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
	snr_list = np.linspace(-20,30,26).astype(np.int32)
	# snr_list = [18]
	data_dic = {}
	for pl,i in enumerate(cls_list):
		for snr in snr_list:
			key = (i,snr)
			data_dic[key] = get_npy(os.path.join(os.getcwd(),'dataset/RML2018a_npy/'+i+'_'+str(snr)+'.npy'))
	return data_dic

def get_pickle_data(file_name):
	if file_name == r'dataset/RML22.pickle.01A':
		file_name = 'D:\pre_work\信噪比控制自监督训练模型\SIT\dataset\RML22.pickle.01A'
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
	elif file_name == r'dataset/RML2016A.pkl':
		file_name = 'D:\pre_work\信噪比控制自监督训练模型\SIT\dataset\RML2016A.pkl'
		with open(file_name, 'rb') as file:
			data = pickle.load(file, encoding='iso-8859-1')
	elif file_name == r'dataset/RML2016b.pkl':
		file_name = 'D:\pre_work\信噪比控制自监督训练模型\SIT\dataset\RML2016b.pkl'
		# data = collect_16b()
		# save_pkl(data,'RML2016b.pkl')
		with open(file_name, 'rb') as file:
			data = pickle.load(file, encoding='iso-8859-1')
	elif file_name == r'dataset/RML2016c.pkl':
		file_name = 'D:\pre_work\信噪比控制自监督训练模型\SIT\dataset\RML2016c.pkl'
		with open(file_name, 'rb') as file:
			data = pickle.load(file, encoding='iso-8859-1')
	elif file_name == r'dataset/RML2018a.pkl':
		file_name = 'D:\pre_work\信噪比控制自监督训练模型\SIT\dataset\RML2018a.pkl'
		# data = collect_18a()
		# save_pkl(data, 'RML2018a.pkl')
		with open(file_name, 'rb') as file:
			data = pickle.load(file, encoding='iso-8859-1')
	return data

def get_SNR_data(data,class_list,SNR):
	data_list = []
	flag = 1
	tem_label = []
	for modulation_cls in class_list:
		if flag == 1:
			tem = np.zeros([0, data[modulation_cls, SNR].shape[1], data[modulation_cls, SNR].shape[2]])
			flag = 0
		tem = np.r_[tem, data[modulation_cls, SNR]]
		tem_label.append(len(data[modulation_cls, SNR]))
	data_list.append(tem)
	return np.stack(data_list,0),tem_label


def mix_data(data,class_list):
	data_list = []
	SNR_list = np.linspace(-20,18,20).astype(np.int32)
	for SNR in SNR_list:
		flag = 1
		tem_label = []
		for modulation_cls in class_list:
			if flag == 1:
				tem = np.zeros([0,data[modulation_cls,SNR].shape[1],data[modulation_cls,SNR].shape[2]])
				flag = 0
			tem = np.r_[tem,data[modulation_cls,SNR]]
			tem_label.append(len(data[modulation_cls,SNR]))
		# data_arr = np.stack(tem, 0)
		data_list.append(tem)
	return np.stack(data_list,0),tem_label


def load_RML_data(dataset='rml16a',SNR=16, norm='maxmin'):
	'''
	:param dataset: select your dataset:rml16a,rml16b,rml18a,rml22
	:SNR : select your SNR or all
	:return: IQ signal array SNR*cls*num*2*128
	'''
	# file_name = 'RML22.pickle.01A'
	if dataset == 'rml22':
		file_name = 'dataset/RML22.pickle.01A'
		class_list = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'WBFM', 'AM-DSB']
	elif dataset == 'rml16a':
		file_name = 'dataset/RML2016A.pkl'
		class_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	elif dataset == 'rml16b':
		file_name = 'dataset/RML2016b.pkl'
		class_list = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	elif dataset == 'rml16c':
		file_name = 'dataset/RML2016c.pkl'
		class_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	elif dataset == 'rml18a':
		file_name = 'dataset/RML2018a.pkl'
		class_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
								  '64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC',
								  'AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
	else:
		file_name = 'RML2016A.pkl'
		class_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	data_dic = get_pickle_data(file_name)
	if SNR == 'all':
		ALL_cls_num_IQ,tem_label = mix_data(data_dic, class_list)
		if dataset == 'rml16c':
			pass
			label = []
			for pl in range(ALL_cls_num_IQ.shape[0]):
				tem = np.zeros([0])
				for pl_j,j in enumerate(tem_label):
					tem = np.r_[tem,np.ones([j])*pl_j]
				label.append(tem)
		else:
			#归一化
			# ALL_cls_num_IQ = normalize_sig(ALL_cls_num_IQ,norm=norm)
			label = []
			for pl in range(ALL_cls_num_IQ.shape[0]):
				tem = np.zeros([0])
				for pl_j,j in enumerate(tem_label):
					tem = np.r_[tem,np.ones([j])*pl_j]
				label.append(tem)
		return ALL_cls_num_IQ,np.stack(label,axis=0)
	else:
		SNR_cls_num_IQ,tem_label = get_SNR_data(data_dic,class_list,SNR)
		# 归一化
		if dataset == 'rml16c':
			pass
			label = []
			for pl in range(SNR_cls_num_IQ.shape[0]):
				tem = np.zeros([0])
				for pl_j,j in enumerate(tem_label):
					tem = np.r_[tem,np.ones([j])*pl_j]
				label.append(tem)
		else:
			# SNR_cls_num_IQ = normalize_sig(SNR_cls_num_IQ, norm=norm)
			# SNR_cls_num_IQ = np.expand_dims(SNR_cls_num_IQ,axis=0)
			label = []
			for pl in range(SNR_cls_num_IQ.shape[0]):
				tem = np.zeros([0])
				for pl_j,j in enumerate(tem_label):
					tem = np.r_[tem,np.ones([j])*pl_j]
				label.append(tem)
		return SNR_cls_num_IQ,np.stack(label,axis=0)

def main():
	# file_name = 'RML22.pickle.01A'
	file_name = 'RML2016c.pkl'
	if file_name == 'RML22.pickle.01A':
		file = 'RML22a'
		class_list = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'WBFM', 'AM-DSB']
	elif file_name == 'RML2016A.pkl':
		file = 'RML16a'
		class_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	elif file_name == 'RML2016B.pkl':
		file = 'RML16b'
		class_list = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	elif file_name == 'RML2016c.pkl':
		file = 'RML16c'
		class_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
	elif file_name == 'RML2018A.pkl':
		file = 'RML18a'
		class_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
								  '64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC',
								  'AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']

	data_dic = get_pickle_data(file_name)
	tar_SNR = 10
	SNR_cls_num_IQ = get_SNR_data(data_dic,class_list,tar_SNR)
	ALL_cls_num_IQ = mix_data(data_dic,class_list)
	SNR_cls_num_IQ = np.expand_dims(SNR_cls_num_IQ,axis=0)


if __name__ == '__main__':
	main()