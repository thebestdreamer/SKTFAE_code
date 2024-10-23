import numpy as np


class RML_cls():
    def __init__(self,select_dataset):
        if select_dataset == 'rml16a':
            self.data_file = 'rml16a'
            self.cls_num = 11
            self.cls_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
            self.SNR = np.linspace(-20,18,20)
            self.num_each_cls_SNR = 1000
            self.length_signal = 128
        elif select_dataset == 'rml16b':
            self.data_file = 'rml16b'
            self.cls_num = 10
            self.cls_list = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
            self.SNR = np.linspace(-20,18,20)
            self.num_each_cls_SNR = 6000
            self.length_signal = 128
        elif select_dataset == 'rml16c':
            self.data_file = 'rml16c'
            self.cls_num = 11
            self.cls_list = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK',
                             'WBFM']
            self.SNR = np.linspace(-20, 18, 20)
            self.num_each_cls_SNR = 'no'
            self.length_signal = 128
        elif select_dataset == 'rml18a':
            self.data_file = 'rml18a'
            self.cls_num = 24
            self.cls_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
								  '64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC',
								  'AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
            self.SNR = np.linspace(-20,30,26)
            self.num_each_cls_SNR = 4096
            self.length_signal = 1024
        elif select_dataset == 'rml22':
            self.data_file = 'rml22'
            self.cls_num = 10
            self.cls_list = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'WBFM', 'AM-DSB']
            self.SNR = np.linspace(-20, 18, 20)
            self.num_each_cls_SNR = 2000
            self.length_signal = 128
