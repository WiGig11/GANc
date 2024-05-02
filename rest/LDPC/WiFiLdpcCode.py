import math
from random import randint
from struct import pack
from tabnanny import check
from turtle import shape
from blinker import receiver_connected
import numpy as np
from WiFiLDPC import H_648_1_2,H_648_2_3,H_648_3_4,H_648_5_6,H_1296_1_2,H_1296_2_3,H_1296_3_4,H_1296_5_6,H_1944_1_2,H_1944_2_3,H_1944_3_4,H_1944_5_6
from utils import rgb2bin,bin2rgb
import tensorflow as tf
class WiFiLDPC():
    def __init__(self, block_length, info_length):
        self._N=block_length
        self._K=info_length
        self._Z=0
        self._M=self._N-self._K
        self._H_mat=np.zeros(shape=(self._N,self._M),dtype=np.uint8)

    def generate_compact_ret(self):
        self._column_mat=[]
        for i in range(self._N):      
            self._column_mat.append([])
        self._row_mat=[]
        for i in range(self._M):      
            self._row_mat.append([])
        for i_col in range(self._N):
            for i_row in range(self._M):
                if self._H_mat[i_col,i_row]==1:
                    self._column_mat[i_col].append(i_row)
                    self._row_mat[i_row].append(i_col)

    def lifted_ldpc(self,baseH):
        self._H_mat=np.zeros(shape=(self._N,self._M),dtype=np.uint8)
        baseH_row,baseH_col=baseH.shape
        for i_base_row in range(baseH_col):
            for i_base_col in range(baseH_row):
                if baseH[i_base_col,i_base_row]>=0:
                    for i_lift in range(self._Z):
                        self._H_mat[self._Z*i_base_col+((i_lift+baseH[i_base_col,i_base_row])%self._Z ), self._Z*i_base_row+i_lift]=1
        self.generate_compact_ret() 

    def load_wifi_ldpc(self,block_length,rate_index):
        
        if block_length==648:
            self._Z=27
        elif block_length==1296:
            self._Z=54
        elif block_length==1944:
            self._Z=81
        else:
            raise ValueError("blocklength should be 648/1296/1944")

        self._N = block_length

        if rate_index==0: 
            # rate 1/2
            self._K=np.uint(self._N/2)
            if block_length==648:
                baseH=np.transpose(H_648_1_2)
            elif block_length==1296:
                baseH=np.transpose(H_1296_1_2)
            elif block_length==1944:
                baseH=np.transpose(H_1944_1_2)
        elif rate_index==1:
            # rate 2/3
            self._K=np.uint(self._N*2/3)
            if block_length==648:
                baseH=np.transpose(H_648_2_3)
            elif block_length==1296:
                baseH=np.transpose(H_1296_2_3)
            elif block_length==1944:
                baseH=np.transpose(H_1944_2_3)
        elif rate_index==2:
            # rate 3/4
            self._K=np.uint(self._N*3/4)
            if block_length==648:
                baseH=np.transpose(H_648_3_4)
            elif block_length==1296:
                baseH=np.transpose(H_1296_3_4)
            elif block_length==1944:
                baseH=np.transpose(H_1944_3_4)
        elif rate_index==3:
            # rate 5/6
            self._K=np.uint(self._N*5/6)
            if block_length==648:
                baseH=np.transpose(H_648_5_6)
            elif block_length==1296:
                baseH=np.transpose(H_1296_5_6)
            elif block_length==1944:
                baseH=np.transpose(H_1944_5_6)
        else:
            raise ValueError("rate should be 1/2 or 2/3 or 3/4 or 5/6")

        self._M = np.uint(self._N - self._K)
        self.lifted_ldpc(baseH)
    
    def check_codeword(self,decoded_cw):
        check=True
        for i_check in range(self._M):
            c=np.uint8(0)
            for i_col_index in range(len(self._row_mat[i_check])):
                i_col=self._row_mat[i_check][i_col_index]
                c=c+decoded_cw[i_col]
            if c%2==1:
                check=False
                break
        return check

    def encode(self,info_bits):
        codeword=np.zeros(shape=(self._N),dtype=np.uint8)
        codeword[:self._K]=info_bits
        parity=np.zeros(shape=(self._M),dtype=np.uint8)

        for i_row in range(self._M):
            for i_col in range(len(self._row_mat[i_row])):
                if self._row_mat[i_row][i_col]<self._K:
                    parity[i_row]+=codeword[self._row_mat[i_row][i_col]]
            parity[i_row]=np.uint8(parity[i_row] % 2)

        for i_col in range(self._Z):
            for i_row in range(i_col,self._M,self._Z):
                codeword[self._K+i_col]+=parity[i_row]
            codeword[self._K+i_col]=np.uint8(codeword[self._K+i_col]%2)

        for i_row in range(self._M):
            for i_col in range(len(self._row_mat[i_row])):
                if (self._row_mat[i_row][i_col]>=self._K) and (self._row_mat[i_row][i_col]<self._K+self._Z):
                    parity[i_row]+=codeword[self._row_mat[i_row][i_col]]
            parity[i_row]=np.uint8(parity[i_row]%2)

        for i_col in range(self._K+self._Z,self._N,self._Z):
            for i_row in range(self._Z):
                codeword[i_col+i_row]=parity[i_col+i_row-self._K-self._Z]
                parity[i_col+i_row-self._K]=np.uint8((parity[i_col+i_row-self._K]+parity[i_col+i_row-self._K-self._Z])%2)
        return codeword

    def decode(self,llr_vec,max_iter,min_sum):
        edge_mat=[]
        last_edge_mat=[]
        for i_row in range(self._M):
            edge_mat.append(np.zeros(shape=(len(self._row_mat[i_row]))))
            last_edge_mat.append(np.zeros(shape=(len(self._row_mat[i_row]))))    
        updated_llr=llr_vec

        decoded_cw=np.zeros(shape=(self._N),dtype=np.uint8)
        for iter in range(max_iter):

            for i_row in range(self._M):
                for i_col_index1 in range(len(self._row_mat[i_row])):
                    temp=1
                    if min_sum:
                        temp=100
                    for i_col_index2 in range(len(self._row_mat[i_row])):
                        if i_col_index1==i_col_index2:
                            continue
                        i_col2=self._row_mat[i_row][i_col_index2]
                        l1=updated_llr[i_col2]-last_edge_mat[i_row][i_col_index2]
                        l1=min(l1,20.0)
                        l1=max(l1,-20.0)
                        if min_sum:
                            sign_temp=1.0
                            if temp<0.0:
                                sign_temp=-1.0
                            sign_l1=1.0
                            if l1<0.0:
                                sign_l1=-1.0
                            temp=sign_temp*sign_l1*min(abs(l1),abs(temp))
                        else:
                            temp=temp*math.tanh(l1/2)
                    if min_sum:
                        edge_mat[i_row][i_col_index1]=temp
                    else:
                        edge_mat[i_row][i_col_index1]=2*math.atanh(temp)

            last_edge_mat=edge_mat
            updated_llr=llr_vec

            for i_row in range(self._M):
                for i_col_index in range(len(self._row_mat[i_row])):
                    i_col=self._row_mat[i_row][i_col_index]
                    updated_llr[i_col]=updated_llr[i_col]+last_edge_mat[i_row][i_col_index]

            for i_col in range(self._N):
                if updated_llr[i_col] > 0 :
                    decoded_cw[i_col]=0
                else:
                    decoded_cw[i_col]=1

            if(self.check_codeword(decoded_cw)):
                break
        return decoded_cw

    def get_message(self,decoded_word):
        return decoded_word[:self._K]

class Constellation():
    def __init__(self, B=2,ave_energy=1,):
        types=["QPSK","16QAM","64QAM"]
        self.type=types[B//2-1]
        '''
        ave_energy== symbol energy
        Gray mapping for QPSK (B=2)

        | b0 |  I  | b1 |  Q  |
        | 0  | -1  | 0  | -1  |
        | 1  |  1  | 1  |  1  |

        Gray mapping for 16-QAM (B=4)

        | b0b1 |  I  | b2b3 |  Q  |
        |  00  | -3  |  00  | -3  |
        |  01  | -1  |  01  | -1  |
        |  11  |  1  |  11  |  1  |
        |  10  |  3  |  10  |  3  |

        Gray mapping for 64-QAM (B=6)

        | b0b1b2 |  I  | b3b4b5 |  Q  |
        |  000   | -7  |  000   | -7  |
        |  001   | -5  |  001   | -5  |
        |  011   | -3  |  011   | -3  |
        |  010   | -1  |  010   | -1  |
        |  110   |  1  |  110   |  1  |
        |  111   |  3  |  111   |  3  |
        |  101   |  5  |  101   |  5  |
        |  100   |  7  |  100   |  7  |
        '''

        self.index = np.arange(2**(B//2))
        
        if B == 2:
            self.map = np.array([-1, 1])
            self.map2 = np.array([[0], [1]])
            self.unit = np.sqrt(ave_energy/2)
        elif B == 4:
            self.map = np.array([-3, -1, 3, 1])
            self.map2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
            self.unit = np.sqrt(ave_energy/10)
        elif B == 6:
            self.map = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
            self.map2 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]])
            self.unit = np.sqrt(ave_energy/42)

        self.inv_map_1 = np.zeros((B//2, 2**(B//2-1)))
        self.inv_map_0 = np.zeros((B//2, 2**(B//2-1)))

        tmp = self.index.copy()
        for i in range(B//2):
            self.inv_map_1[i] = np.where(tmp>=2**(B//2-1-i))[0] 
            self.inv_map_0[i] = np.where(tmp<2**(B//2-1-i))[0] 
            tmp[tmp>=2**(B//2-1-i)] -= 2**(B//2-1-i)
        
        self.bound = self.unit*(np.arange(-2**(B//2)+2, 2**(B//2), 2))
        self.B = B

    def llr_compute(self, y, snr, simpler=False):
        # replace gauss distribution with other distribution?
        sigma = 10 ** (-snr / 20) /np.sqrt(2)
        M = y.shape[0]
        LLR = []
        for m in range(M):
            sym = y[m]
            f = np.vectorize(int)
            LLR_real = []
            LLR_imag = []

            for i in range(self.B//2):
                pos_0 = self.inv_map_0[i]
                pos_1 = self.inv_map_1[i]
                sym_0 = self.map[f(pos_0)]*self.unit
                sym_1 = self.map[f(pos_1)]*self.unit
                if not simpler:
                    prob_0_real = np.sum(np.exp(-(sym.real - sym_0)**2/(sigma**2)))
                    prob_1_real = np.sum(np.exp(-(sym.real - sym_1)**2/(sigma**2)))

                    prob_0_imag = np.sum(np.exp(-(sym.imag - sym_0)**2/(sigma**2)))
                    prob_1_imag = np.sum(np.exp(-(sym.imag - sym_1)**2/(sigma**2)))

                    ratio_real = np.log(prob_0_real) - np.log(prob_1_real+np.spacing(1))
                    ratio_imag = np.log(prob_0_imag) - np.log(prob_1_imag+np.spacing(1))
                else:
                    """
                    prob_0_real = np.min(abs(sym.real - sym_0)**2)
                    prob_1_real = np.min(abs(sym.real - sym_1)**2)
                    """
                    prob_0_real = np.max(np.abs((sym.real - sym_0)**2))
                    prob_1_real = np.max(np.abs((sym.real - sym_1)**2))

                    prob_0_imag = np.max(np.abs((sym.imag - sym_0)**2))
                    prob_1_imag = np.max(np.abs((sym.imag - sym_1)**2))

                    ratio_real = (prob_0_real-prob_1_real)/(sigma**2)
                    ratio_imag = (prob_0_imag-prob_1_imag)/(sigma**2)

                LLR_real.append(ratio_real)
                LLR_imag.append(ratio_imag)

            LLR.append(np.stack(LLR_real))
            LLR.append(np.stack(LLR_imag))
            
        return np.hstack(LLR)

    def modulate(self, x):
        '''
        input: Nx1
        '''
        tx = x.reshape(x.shape[0]//self.B, self.B)
        tx_I = tx[:, :self.B//2]
        tx_Q = tx[:, self.B//2:]
        index_I = np.zeros(tx_I.shape[0])
        index_Q = np.zeros(tx_Q.shape[0])

        for i in range(self.B//2):
            index_I += 2**(self.B//2-i-1)*tx_I[:,i]
            index_Q += 2**(self.B//2-i-1)*tx_Q[:,i]

        f = np.vectorize(int)
        tx_sym = self.unit*(self.map[f(index_I)] + self.map[f(index_Q)]*1j)
        return tx_sym


    def demodulate(self, y):
        '''
        input: N/Bx1
        '''
        M = y.shape[0]
        code_list = []
        for m in range(M):
            sym = y[m]
            
            index_real = np.where(self.bound>sym.real)[0]
            if index_real.shape[0] == 0:
                pos_real = 2**(self.B//2)-1
            else:
                pos_real = np.min(index_real)

            code_list.append(self.map2[int(pos_real)])

            index_imag = np.where(self.bound>sym.imag)[0]
            if index_imag.shape[0] == 0:
                pos_imag = 2**(self.B//2)-1
            else:
                pos_imag = np.min(index_imag)

            code_list.append(self.map2[int(pos_imag)])

        return np.hstack(code_list)
class Fading():
    """
    H == CSI
    K == Rice factor == P_LOS / P_Rayleigh, denotes line-of-sight component strength
    
    """
    def __init__(self,snr, channel_type="AWGN", h=None,k=None):
        self.channel_type=channel_type
        self.h=h
        self.k=k
        self.sigma = np.sqrt(10 ** (-snr / 10))

    def addFading(self,x):
        if(self.channel_type=="AWGN"):
            noise = self.sigma /np.sqrt(2) * (np.random.randn(x.shape[0]) + 1j*np.random.randn(x.shape[0]))
            h=np.ones_like(x)+1j*np.zeros_like(x)
            y = x + noise
        elif (self.channel_type=="Rayleigh"):
            if self.h is None:
                h =  (np.random.randn(x.shape[0]) + 1j*np.random.randn(x.shape[0]))/np.sqrt(2)
            noise = self.sigma  /np.sqrt(2) * (np.random.randn(x.shape[0]) + 1j*np.random.randn(x.shape[0]))
            
            y = h*x + noise
            
        elif (self.channel_type=="Rice"):
            if(self.k==None):
                raise ValueError("expected factor k in Rice channel")
            if self.h is None:
                h = (np.random.randn(x.shape[0]) + 1j*np.random.randn(x.shape[0]))/np.sqrt(2)
                h=np.sqrt(self.k/self.k+1)+np.sqrt(1/self.k+1)*h
            noise = self.sigma /np.sqrt(2)* (np.random.randn(x.shape[0]) + 1j*np.random.randn(x.shape[0]))
            y = h*x + noise
        return y,h


    def deFading(self,y,h):
        if(self.channel_type=="AWGN"):
            y_chn_eq=y
        elif (self.channel_type=="Rayleigh"):
            #Perfect Channel Estimation
            y_chn_eq=y*np.conj(h)/(np.abs(h)**2)
            #y_chn_eq=y/h
        elif (self.channel_type=="Rice"):
            #Channel Estimation
            y_chn_eq=y
        return y_chn_eq
        
def save_img(image,snr,mod,blocklength,rate,channal):
    image=tf.math.multiply(image,255)
    image=tf.cast(image,dtype=tf.uint8)
    image=tf.image.encode_png(image)
    with tf.io.gfile.GFile('./SemCom/GAN/img_kodak/image_L{}_R{}_M{}_{}in{}.png'.format(blocklength,rate,mod,snr,channal), 'wb') as file:
        file.write(image.numpy())
    print("image have saved")

"""
# array test

soft_ber_awgn=[]
soft_ber_rayl=[]
hard_ber_awgn=[]
hard_ber_rayl=[]

blockLengths=[648,1296,1944]
rates=[0,1,2,3]
snrs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
Bits=[2,4,6]

ldpc_instance=WiFiLDPC(0,0)
for blockLength in blockLengths:
    for rate in rates:
        ldpc_instance.load_wifi_ldpc(blockLength,rate)

        info_bits=np.random.randint(2,size=ldpc_instance._K)

        for Bit in Bits:
            modulate_instance=Constellation(B=Bit)
            print("modulation==",modulate_instance.type)

            bers_awgn_soft=[]
            bers_awgn_hard=[]
            bers_rayl_soft=[]
            bers_rayl_hard=[]

            for snr in snrs:
                
                awgn=Fading(snr,channel_type="AWGN")
                rayleigh=Fading(snr,channel_type="Rayleigh")

                code_word=ldpc_instance.encode(info_bits)
                mod_sym=modulate_instance.modulate(code_word)
        
                receive_word,h=awgn.addFading(mod_sym)
                receive_word=awgn.deFading(receive_word,h)

                llr_vec=modulate_instance.llr_compute(receive_word,snr)
                newMsg_awgn_soft=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                ber_awgn_soft=abs(newMsg_awgn_soft - info_bits).sum()
                bers_awgn_soft.append(ber_awgn_soft)
        
                demod_sym=modulate_instance.demodulate(receive_word)
                llr_vec = np.zeros(demod_sym.shape)
                llr_vec[demod_sym == 0] = 9
                llr_vec[demod_sym == 1] = -9
                newMsg_awgn_hard=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                ber_awgn_hard=abs(newMsg_awgn_hard - info_bits).sum()
                bers_awgn_hard.append(ber_awgn_hard)
        
                receive_word,h=rayleigh.addFading(mod_sym)
                receive_word=rayleigh.deFading(receive_word,h)
                llr_vec=modulate_instance.llr_compute(receive_word,snr)
                newMsg_rayl_soft=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                ber_rayl_soft=abs(newMsg_rayl_soft - info_bits).sum()
                bers_rayl_soft.append(ber_rayl_soft)

                demod_sym=modulate_instance.demodulate(receive_word)
                llr_vec = np.zeros(demod_sym.shape)
                llr_vec[demod_sym == 0] = 9
                llr_vec[demod_sym == 1] = -9
                newMsg_rayl_hard=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                ber_rayl_hard=abs(newMsg_rayl_hard - info_bits).sum()
                bers_rayl_hard.append(ber_rayl_hard)

            print("L{}_R{}_M{}_ber_awgn_soft==".format(blockLength,rate,modulate_instance.type),bers_awgn_soft)
            print("L{}_R{}_M{}_ber_awgn_hard==".format(blockLength,rate,modulate_instance.type),bers_awgn_hard)
            print("L{}_R{}_M{}_ber_rayl_soft==".format(blockLength,rate,modulate_instance.type),bers_rayl_soft)
            print("L{}_R{}_M{}_ber_rayl_hard==".format(blockLength,rate,modulate_instance.type),bers_rayl_hard)
    
"""
"""
#image test
example=WiFiLDPC(0,0)
example.load_wifi_ldpc(648,0)
#info_bits=np.random.randint(2,size=example._K)
img=np.random.randint(256,size=(128,256,3),dtype=np.uint8)
img_bin=rgb2bin(img)
img_bin_flatten=img_bin.flatten()
n_blocks=img_bin_flatten.size // example._K
residual=img_bin_flatten.size % example._K
if residual:
    n_blocks+=1
packet=np.zeros(example._K*n_blocks,dtype=np.int32)
packet[:img_bin_flatten.size]=img_bin_flatten
total_bits=example._K*n_blocks
print(total_bits)
packet=packet.reshape(n_blocks,example._K)

snrs=[0,2,4,6,8,10,12,14,16,18,20]
qpsk=Constellation(B=4)

ber_soft_awgn=[]
ber_soft_rayl=[]
ber_hard_awgn=[]
ber_hard_rayl=[]
for snr in snrs:
    print("snr==",snr)
    awgn=Fading(snr,channel_type="AWGN")
    rayleigh=Fading(snr,channel_type="Rayleigh")
    
    ber_soft_awgn_total=0
    ber_soft_rayl_total=0
    ber_hard_awgn_total=0
    ber_hard_rayl_total=0

    for curblock in range(n_blocks):
        info_bits=packet[curblock,:]
        code_word=example.encode(info_bits)
        mod_sym=qpsk.modulate(code_word)
        
        receive_word,h=awgn.addFading(mod_sym)
        receive_word=awgn.deFading(receive_word,h)

        llr_vec=qpsk.llr_compute(receive_word,snr)
        soft_awgn=example.get_message(example.decode(llr_vec,20,False))
        ber_soft_awgn_total+=abs(soft_awgn - info_bits).sum()
        
        demod_sym=qpsk.demodulate(receive_word)
        llr_vec = np.zeros(demod_sym.shape)
        llr_vec[demod_sym == 0] = 9
        llr_vec[demod_sym == 1] = -9
        hard_awgn=example.get_message(example.decode(llr_vec,20,False))
        ber_hard_awgn_total+=abs(hard_awgn - info_bits).sum()
        
        receive_word,h=rayleigh.addFading(mod_sym)
        receive_word=rayleigh.deFading(receive_word,h)
        llr_vec=qpsk.llr_compute(receive_word,snr)
        soft_rayl=example.get_message(example.decode(llr_vec,20,False))
        ber_soft_rayl_total+=abs(soft_rayl - info_bits).sum()

        demod_sym=qpsk.demodulate(receive_word)
        llr_vec = np.zeros(demod_sym.shape)
        llr_vec[demod_sym == 0] = 9
        llr_vec[demod_sym == 1] = -9
        hard_rayl=example.get_message(example.decode(llr_vec,20,False))
        ber_hard_rayl_total+=abs(hard_rayl - info_bits).sum()

    ber_soft_awgn.append(ber_soft_awgn_total)
    ber_hard_awgn.append(ber_hard_awgn_total)
    ber_soft_rayl.append(ber_soft_rayl_total)
    ber_hard_rayl.append(ber_hard_rayl_total)

print("soft_awgn==",ber_soft_awgn)
print("soft_rayl==",ber_hard_awgn)
print("hard_awgn==",ber_soft_rayl)
print("hard_rayl==",ber_hard_rayl)
"""
"""
print(example._H_mat.shape)
example.load_wifi_ldpc(648,1)
print(example._H_mat.shape)
example.load_wifi_ldpc(648,2)
print(example._H_mat.shape)
example.load_wifi_ldpc(648,3)
print(example._H_mat.shape)
example.load_wifi_ldpc(1296,0)
print(example._H_mat.shape)
example.load_wifi_ldpc(1296,1)
print(example._H_mat.shape)
example.load_wifi_ldpc(1296,2)
print(example._H_mat.shape)
example.load_wifi_ldpc(1296,3)
print(example._H_mat.shape)
example.load_wifi_ldpc(1944,0)
print(example._H_mat.shape)
example.load_wifi_ldpc(1944,1)
print(example._H_mat.shape)
example.load_wifi_ldpc(1944,2)
print(example._H_mat.shape)
example.load_wifi_ldpc(1944,3)
print(example._H_mat.shape)     
"""      
"""
from PIL import Image
img_ori=np.asarray(Image.open(r'H:\DLCode\myGAN\img\ori.png').convert("RGB"))
img_C8=np.asarray(Image.open(r'H:\DLCode\myGAN\img\SemComC8\image_at_epoch_1000.png').convert("RGB"))
img_C8_bin=rgb2bin(img_C8)
img_C8_bin_flatten=img_C8_bin.flatten()

blockLengths=[648,1296,1944]
rates=[0,1,2,3]
snrs=[0,2,4,6,8,10,12,14,16,18,20]
Bits=[2,4,6]

ldpc_instance=WiFiLDPC(0,0)
for blockLength in blockLengths:
    for rate in rates:
        ldpc_instance.load_wifi_ldpc(blockLength,rate)

        n_blocks=img_C8_bin_flatten.size // ldpc_instance._K
        residual=img_C8_bin_flatten.size % ldpc_instance._K
        if residual:
            n_blocks+=1
        packet=np.zeros(ldpc_instance._K*n_blocks,dtype=np.int32)
        packet[:img_C8_bin_flatten.size]=img_C8_bin_flatten
        total_bits=ldpc_instance._K*n_blocks
        print(total_bits)
        packet=packet.reshape(n_blocks,ldpc_instance._K)

        for Bit in Bits:
            modulate_instance=Constellation(B=Bit)
            print("modulation==",modulate_instance.type)
            ber_soft_awgn=[]
            PSNRs_awgn_soft=[]
            for snr in snrs:
                print("snr==",snr)
                awgn=Fading(snr,channel_type="AWGN")
                newMsg_awgn_soft=np.zeros_like(packet)
                ber_soft_awgn_total=0
                for curblock in range(n_blocks):
                    info_bits=packet[curblock,:]
                    code_word=ldpc_instance.encode(info_bits)
                    mod_sym=modulate_instance.modulate(code_word)
        
                    receive_word,h=awgn.addFading(mod_sym)
                    receive_word=awgn.deFading(receive_word,h)

                    llr_vec=modulate_instance.llr_compute(receive_word,snr)
                    soft_awgn=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                    ber_soft_awgn_total+=abs(soft_awgn - info_bits).sum()
                    newMsg_awgn_soft[curblock,:]=soft_awgn
        
                ber_soft_awgn.append(ber_soft_awgn_total)
                newMsg_awgn_soft=newMsg_awgn_soft.flatten()[:img_C8_bin_flatten.size]
                newMsg_awgn_soft = newMsg_awgn_soft.reshape(img_C8_bin.shape)
                img_awgn_soft=bin2rgb(newMsg_awgn_soft)
                PSNR_awgn_soft=tf.image.psnr(img_ori,img_awgn_soft,max_val=255)
                SSIM_awgn_soft=tf.image.ssim(img_ori,img_awgn_soft,max_val=255)
                MSSSIM_awgn_soft=tf.image.ssim_multiscale(img_ori,img_awgn_soft,max_val=255)
                PSNRs_awgn_soft.append(PSNR_awgn_soft)

            print("soft_awgn_{}_{}_{}==".format(blockLength,rate,modulate_instance.type),ber_soft_awgn)
"""     


#final version
from PIL import Image
img_ori=np.asarray(Image.open(r'H:\GradDesign\paper\ori.png').convert("RGB"))
img_C8=np.asarray(Image.open(r'H:\GradDesign\paper\C8.png').convert("RGB"))
img_C8_bin=rgb2bin(img_C8)
img_C8_bin_flatten=img_C8_bin.flatten()

#blockLengths=[648,1296,1944]
#rates=[0,1,2,3]
#snrs=[0,4,8,12,16,20]
#Bits=[2,4,6]
blockLengths=[1296]
rates=[0,1]
snrs=[0,2,4,6,8,10,12,14,16,18,20]
Bits=[2,4,6]

ldpc_instance=WiFiLDPC(0,0)
for blockLength in blockLengths:
    for rate in rates:
        ldpc_instance.load_wifi_ldpc(blockLength,rate)

        n_blocks=img_C8_bin_flatten.size // ldpc_instance._K
        residual=img_C8_bin_flatten.size % ldpc_instance._K
        if residual:
            n_blocks+=1
        packet=np.zeros(ldpc_instance._K*n_blocks,dtype=np.int32)
        packet[:img_C8_bin_flatten.size]=img_C8_bin_flatten
        total_bits=ldpc_instance._K*n_blocks
        print(total_bits)
        packet=packet.reshape(n_blocks,ldpc_instance._K)

        for Bit in Bits:
            modulate_instance=Constellation(B=Bit)
            print("modulation==",modulate_instance.type)

            ber_awgn_soft=[]
            ber_rayl_soft=[]
            ber_awgn_hard=[]
            ber_rayl_hard=[]

            PSNRs_awgn_soft=[]
            SSIMs_awgn_soft=[]
            MSSSIMs_awgn_soft=[]
            PSNRs_awgn_hard=[]
            SSIMs_awgn_hard=[]
            MSSSIMs_awgn_hard=[]

            PSNRs_rayl_soft=[]
            SSIMs_rayl_soft=[]
            MSSSIMs_rayl_soft=[]
            PSNRs_rayl_hard=[]
            SSIMs_rayl_hard=[]
            MSSSIMs_rayl_hard=[]

            for snr in snrs:
                print("snr==",snr)
                awgn=Fading(snr,channel_type="AWGN")
                rayleigh=Fading(snr,channel_type="Rayleigh")
    
                ber_awgn_soft_total=0
                ber_rayl_soft_total=0
                ber_awgn_hard_total=0
                ber_rayl_hard_total=0

                newMsg_awgn_soft=np.zeros_like(packet)
                newMsg_awgn_hard=np.zeros_like(packet)
                newMsg_rayl_soft=np.zeros_like(packet)
                newMsg_rayl_hard=np.zeros_like(packet)

                for curblock in range(n_blocks):
                    info_bits=packet[curblock,:]
                    code_word=ldpc_instance.encode(info_bits)
                    mod_sym=modulate_instance.modulate(code_word)
        
                    receive_word,h=awgn.addFading(mod_sym)
                    receive_word=awgn.deFading(receive_word,h)

                    llr_vec=modulate_instance.llr_compute(receive_word,snr)
                    newMsg_awgn_soft[curblock,:]=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                    ber_awgn_soft_total+=abs(newMsg_awgn_soft[curblock,:] - info_bits).sum()
        
                    demod_sym=modulate_instance.demodulate(receive_word)
                    llr_vec = np.zeros(demod_sym.shape)
                    llr_vec[demod_sym == 0] = 9
                    llr_vec[demod_sym == 1] = -9
                    newMsg_awgn_hard[curblock,:]=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                    ber_awgn_hard_total+=abs(newMsg_awgn_hard[curblock,:] - info_bits).sum()
        
                    receive_word,h=rayleigh.addFading(mod_sym)
                    receive_word=rayleigh.deFading(receive_word,h)
                    llr_vec=modulate_instance.llr_compute(receive_word,snr)
                    newMsg_rayl_soft[curblock,:]=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                    ber_rayl_soft_total+=abs(newMsg_rayl_soft[curblock,:] - info_bits).sum()

                    demod_sym=modulate_instance.demodulate(receive_word)
                    llr_vec = np.zeros(demod_sym.shape)
                    llr_vec[demod_sym == 0] = 9
                    llr_vec[demod_sym == 1] = -9
                    newMsg_rayl_hard[curblock,:]=ldpc_instance.get_message(ldpc_instance.decode(llr_vec,20,False))
                    ber_rayl_hard_total+=abs(newMsg_rayl_hard[curblock,:] - info_bits).sum()

                ber_awgn_soft.append(ber_awgn_soft_total)
                ber_awgn_hard.append(ber_awgn_hard_total)
                ber_rayl_soft.append(ber_rayl_soft_total)
                ber_rayl_hard.append(ber_rayl_hard_total)

                newMsg_awgn_soft=newMsg_awgn_soft.flatten()[:img_C8_bin_flatten.size]
                newMsg_awgn_soft = newMsg_awgn_soft.reshape(img_C8_bin.shape)
                img_awgn_soft=bin2rgb(newMsg_awgn_soft)
                save_img(img_awgn_soft,snr,modulate_instance.type,blockLength,rate,"awgn_soft")
                PSNR_awgn_soft=tf.image.psnr(img_ori,img_awgn_soft,max_val=255)
                SSIM_awgn_soft=tf.image.ssim(img_ori,img_awgn_soft,max_val=255)
                MSSSIM_awgn_soft=tf.image.ssim_multiscale(img_ori,img_awgn_soft,max_val=255)

                newMsg_awgn_hard=newMsg_awgn_hard.flatten()[:img_C8_bin_flatten.size]
                newMsg_awgn_hard = newMsg_awgn_hard.reshape(img_C8_bin.shape)
                img_awgn_hard=bin2rgb(newMsg_awgn_hard)
                save_img(img_awgn_soft,snr,modulate_instance.type,blockLength,rate,"awgn_hard")
                PSNR_awgn_hard=tf.image.psnr(img_ori,img_awgn_hard,max_val=255)
                SSIM_awgn_hard=tf.image.ssim(img_ori,img_awgn_hard,max_val=255)
                MSSSIM_awgn_hard=tf.image.ssim_multiscale(img_ori,img_awgn_hard,max_val=255)
                
                newMsg_rayl_soft=newMsg_rayl_soft.flatten()[:img_C8_bin_flatten.size]
                newMsg_rayl_soft = newMsg_rayl_soft.reshape(img_C8_bin.shape)
                img_rayl_soft=bin2rgb(newMsg_rayl_soft)
                save_img(img_awgn_soft,snr,modulate_instance.type,blockLength,rate,"rayl_soft")
                PSNR_rayl_soft=tf.image.psnr(img_ori,img_rayl_soft,max_val=255)
                SSIM_rayl_soft=tf.image.ssim(img_ori,img_rayl_soft,max_val=255)
                MSSSIM_rayl_soft=tf.image.ssim_multiscale(img_ori,img_rayl_soft,max_val=255)

                newMsg_rayl_hard=newMsg_rayl_hard.flatten()[:img_C8_bin_flatten.size]
                newMsg_rayl_hard = newMsg_rayl_hard.reshape(img_C8_bin.shape)
                img_rayl_hard=bin2rgb(newMsg_rayl_hard)
                save_img(img_awgn_soft,snr,modulate_instance.type,blockLength,rate,"rayl_hard")
                PSNR_rayl_hard=tf.image.psnr(img_ori,img_rayl_hard,max_val=255)
                SSIM_rayl_hard=tf.image.ssim(img_ori,img_rayl_hard,max_val=255)
                MSSSIM_rayl_hard=tf.image.ssim_multiscale(img_ori,img_rayl_hard,max_val=255)


                PSNRs_awgn_soft.append(PSNR_awgn_soft)
                SSIMs_awgn_soft.append(SSIM_awgn_soft)
                MSSSIMs_awgn_soft.append(MSSSIM_awgn_soft)

                PSNRs_awgn_hard.append(PSNR_awgn_hard)
                SSIMs_awgn_hard.append(SSIM_awgn_hard)
                MSSSIMs_awgn_hard.append(MSSSIM_awgn_hard)

                PSNRs_rayl_soft.append(PSNR_rayl_soft)
                SSIMs_rayl_soft.append(SSIM_rayl_soft)
                MSSSIMs_rayl_soft.append(MSSSIM_rayl_soft)
                
                PSNRs_rayl_hard.append(PSNR_rayl_hard)
                SSIMs_rayl_hard.append(SSIM_rayl_hard)
                MSSSIMs_rayl_hard.append(MSSSIM_rayl_hard)

            print("ber_awgn_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),ber_awgn_soft)
            print("ber_awgn_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),ber_awgn_hard)
            print("ber_rayl_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),ber_rayl_soft)
            print("ber_rayl_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),ber_rayl_hard)

            print("PSNR_awgn_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),PSNRs_awgn_soft)
            print("SSIM_awgn_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),SSIMs_awgn_soft)
            print("MSSSIM_awgn_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),MSSSIMs_awgn_soft)
            print("PSNR_awgn_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),PSNRs_awgn_hard)
            print("SSIM_awgn_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),SSIMs_awgn_hard)
            print("MSSSIM_awgn_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),MSSSIMs_awgn_hard)
            print("PSNR_rayl_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),PSNRs_rayl_soft)
            print("SSIM_rayl_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),SSIMs_rayl_soft)
            print("MSSSIM_rayl_soft_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),MSSSIMs_rayl_soft)
            print("PSNR_rayl_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),PSNRs_rayl_hard)
            print("SSIM_rayl_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),SSIMs_rayl_hard)
            print("MSSSIM_rayl_hard_N{}_R{}_M{}==".format(blockLength,rate,modulate_instance.type),MSSSIMs_rayl_hard)

            np.savetxt(fname="./LDPCandFading/BPG_ber_awgn_soft_N{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=ber_awgn_soft,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_ber_awgn_hard_N{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=ber_awgn_hard,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_ber_rayl_soft_N{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=ber_rayl_soft,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_ber_rayl_hard_N{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=ber_rayl_hard,fmt="%f")


            np.savetxt(fname="./LDPCandFading/BPG_PSNR_awgn_soft{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=PSNRs_awgn_soft,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_SSIM_awgn_soft{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=SSIMs_awgn_soft,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_MSSSIM_awgn_soft{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=MSSSIMs_awgn_soft,fmt="%f")

            np.savetxt(fname="./LDPCandFading/BPG_PSNR_awgn_hard{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=PSNRs_awgn_hard,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_SSIM_awgn_hard{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=SSIMs_awgn_hard,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_MSSSIM_awgn_hard{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=MSSSIMs_awgn_hard,fmt="%f")

            np.savetxt(fname="./LDPCandFading/BPG_PSNR_rayl_soft{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=PSNRs_rayl_soft,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_SSIM_rayl_soft{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=SSIMs_rayl_soft,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_MSSSIM_rayl_soft{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=MSSSIMs_rayl_soft,fmt="%f")

            np.savetxt(fname="./LDPCandFading/BPG_PSNR_rayl_hard{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=PSNRs_rayl_hard,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_SSIM_rayl_hard{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=SSIMs_rayl_hard,fmt="%f")
            np.savetxt(fname="./LDPCandFading/BPG_MSSSIM_rayl_hard{}_R{}_M{}.csv".format(blockLength,rate,modulate_instance.type),X=MSSSIMs_rayl_hard,fmt="%f")
