import tensorflow as tf
import os
import numpy as np
import pdb
from PIL import Image

def get_GANs_metric():
    oriimagedir = ''
    GANsPPSNRs_all = []
    GANsPSSIMs_all = []
    for bottleneck in C:
        datadir = ''
        resimagedir = os.path.join(datadir,str(bottleneck))
        files = os.listdir(resimagedir)
        for file in files:
            origilename = os.splitext('_')
            origile = os.path.join(oriimagedir,origilename)
            GANsPPSNRs = GANsPPSNRs+tf.image.psnr(file,origile)
            GANsPSSIMs = GANsPSSIMs+tf.image.ssim_multiscale(file,origile)
        GANsPPSNRs = np.mean(GANsPPSNRs)
        GANsPSSIMs = np.mean(GANsPSSIMs)
        GANsPPSNRs_all.append(GANsPPSNRs)
        GANsPSSIMs_all.append(GANsPSSIMs)
    return GANsPPSNRs_all,GANsPSSIMs_all

def get_GANs_metric2():
    oriimagedir = 'res/test_res/testkodak1/ori/'
    resimagedir = 'res/test_res/testkodak1/res/'
    GANsPPSNRs_all = []
    GANsPSSIMs_all = []
    files = os.listdir(oriimagedir)
    for file in files:
        orifilename = oriimagedir+str(file)
        parts = file.split("img")
        #pdb.set_trace()
        content = parts[1].split(".")[0]
        resfilename = 'single_test'+str(content)+'.png'
        resfile = os.path.join(resimagedir,resfilename)
        print("testing original image:",orifilename,"testing test image:",resfile)
        orifilename=np.asarray(Image.open(orifilename).convert("RGB"))
        resfile=np.asarray(Image.open(resfile).convert("RGB"))
        #
        GANsPPSNRs_all.append(tf.image.psnr(orifilename,resfile,max_val=255))
        GANsPSSIMs_all.append(tf.image.ssim_multiscale(orifilename,resfile,max_val=255))
    return GANsPPSNRs_all,GANsPSSIMs_all

def get_single_GANs_metric2():
    orifilename=np.asarray(Image.open('/chz/kodak/kodim01.png').convert("RGB"))
    resfile=np.asarray(Image.open('/chz/res/test_res/testkodak1/res/single_testkodim01.png').convert("RGB"))
    psnr = tf.image.psnr(orifilename,resfile,max_val=255)
    ssim = tf.image.ssim_multiscale(orifilename,resfile,max_val=255)
    print(psnr)
    print(ssim)


get_single_GANs_metric2()
'''
C = [2,4,6,8]
GANsBPPs = []
for i in range(len(C)):
    GANsBPPs.append(2.321928*C[i]/(16*16))
#pdb.set_trace()
GANsPPSNRs_all,GANsPSSIMs_all = get_GANs_metric2()
print(GANsPPSNRs_all)
print(np.max(GANsPPSNRs_all))
print(np.mean(GANsPPSNRs_all))
print(np.min(GANsPPSNRs_all))
print(GANsPSSIMs_all)
print(np.max(GANsPSSIMs_all))
print(np.mean(GANsPSSIMs_all))
print(np.min(GANsPSSIMs_all))

'''