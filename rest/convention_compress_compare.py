import tensorflow as tf
import os
from PIL import Image
import numpy as np


data_dir='H:\GradDesign\dataset\kodak'
#data_dir='H:\GradDesign\dataset\city\img2'

curDir=os.getcwd()
bpg_dir=os.path.join(curDir,r'convention_compression\bpg-0.9.8-win64')
webP_dir=os.path.join(curDir,r'convention_compression\libwebp-1.3.0-windows-x64\bin')
jpeg_dir=os.path.join(curDir,r'convention_compression\jpeg-9c')
jpeg2k_dir=os.path.join(curDir,r'convention_compression\openjpeg-v2.5.0-windows-x64\bin')
print(curDir)
print(bpg_dir)
print(webP_dir)
print(jpeg_dir)
print(jpeg2k_dir)

BPGBPPs=[]
BPGPSNRs=[]
BPGSSIMs=[]
BPGMSSSIMs=[]
# BPG
os.chdir(bpg_dir)
save_enc_dir = os.path.join(bpg_dir,r'encImg_bpg')   
save_dec_dir = os.path.join(bpg_dir,r'decImg_bpg')

if not os.path.exists(save_enc_dir):
    os.makedirs(save_enc_dir)
if not os.path.exists(save_dec_dir):
    os.makedirs(save_dec_dir)

for curDir,dirs,files in os.walk(data_dir):
    for file in files:
        filename,postfix=os.path.splitext(file)
        file_dir=os.path.join(curDir,file)
        
        if filename!="kodim01":
            continue
        #print("filename==",filename)
        #print("file_dir==",file_dir)
        originImg=np.asarray(Image.open(file_dir).convert("RGB"))
        height,width,channel=originImg.shape
        for q in range(29,52):
            #print("q==",q)
            ImgQ_dir=os.path.join(save_dec_dir,filename + 'q{}.png'.format(q))
            #print("ImgQ_dir==",ImgQ_dir)
            bpgfile_dir=os.path.join(save_enc_dir,filename + 'q{}.bpg'.format(q))
            #print("bpgfile_dir==",bpgfile_dir)
            os.system('bpgenc -m 9 -b 8 -q {}'.format(q) + '  ' + file_dir + ' -o ' + bpgfile_dir) # -m 控制速度 -b 控制位深度 -q 控制图像质量 -o 输出图像
            os.system('bpgdec -o ' + ImgQ_dir + ' ' + bpgfile_dir)
            ImgQ=np.asarray(Image.open(ImgQ_dir).convert("RGB"))
            fsize = os.path.getsize(bpgfile_dir)
            bpp=fsize*8/height/width
            #print("bpp==",bpp)
            psnr=tf.image.psnr(originImg,ImgQ,max_val=255)
            #print(psnr)
            ssim=tf.image.ssim(originImg,ImgQ,max_val=255)
            #print(ssim)
            msssim=tf.image.ssim_multiscale(originImg,ImgQ,max_val=255)
            #print(msssim)
            BPGBPPs.append(bpp)
            BPGPSNRs.append(psnr.numpy())
            BPGSSIMs.append(ssim.numpy())
            BPGMSSSIMs.append(msssim.numpy())



WebPBPPs=[]
WebPPSNRs=[]
WebPSSIMs=[]
WebPMSSSIMs=[]
# webP
os.chdir(webP_dir)
save_enc_dir = os.path.join(webP_dir,r'encImg_webP')   
save_dec_dir = os.path.join(webP_dir,r'decImg_webP')

if not os.path.exists(save_enc_dir):
    os.makedirs(save_enc_dir)
if not os.path.exists(save_dec_dir):
    os.makedirs(save_dec_dir)


for curDir,dirs,files in os.walk(data_dir):
    for file in files:
        filename,postfix=os.path.splitext(file)
        file_dir=os.path.join(curDir,file)
        if filename!="kodim01":
            continue
        #print("filename==",filename)
        #print("file_dir==",file_dir)
        
        originImg=np.asarray(Image.open(file_dir).convert("RGB"))
        height,width,channel=originImg.shape
        for q in range(1,30):
            #print("q==",q)
            ImgQ_dir=os.path.join(save_dec_dir,filename + 'q{}.png'.format(q))
            #print("ImgQ_dir==",ImgQ_dir)
            webPfile_dir=os.path.join(save_enc_dir,filename + 'q{}.webp'.format(q))
            #print("webPfile_dir==",webPfile_dir)
            os.system('cwebp -q {}'.format(q) + '  ' + file_dir + ' -o ' + webPfile_dir)
            os.system('dwebp ' + webPfile_dir + ' -o ' + ImgQ_dir)
     
            ImgQ=np.asarray(Image.open(ImgQ_dir).convert("RGB"))
            fsize = os.path.getsize(webPfile_dir)
            bpp=fsize*8/height/width
            #print("bpp==",bpp)
            psnr=tf.image.psnr(originImg,ImgQ,max_val=255)
            #print(psnr)
            ssim=tf.image.ssim(originImg,ImgQ,max_val=255)
            #print(ssim)
            msssim=tf.image.ssim_multiscale(originImg,ImgQ,max_val=255)
            #print(msssim)
            WebPBPPs.append(bpp)
            WebPPSNRs.append(psnr.numpy())
            WebPSSIMs.append(ssim.numpy())
            WebPMSSSIMs.append(msssim.numpy())


J2KBPPs=[]
J2KPSNRs=[]
J2KSSIMs=[]
J2KMSSSIMs=[]
# JPEG2000
os.chdir(jpeg2k_dir)
save_enc_dir = os.path.join(jpeg2k_dir,r'encImg_JPEG2K')   
save_dec_dir = os.path.join(jpeg2k_dir,r'decImg_JPEG2K')
data_dir='H:\GradDesign\dataset\kodak_bmp'
if not os.path.exists(save_enc_dir):
    os.makedirs(save_enc_dir)
if not os.path.exists(save_dec_dir):
    os.makedirs(save_dec_dir)


for curDir,dirs,files in os.walk(data_dir):
    for file in files:
        filename,postfix=os.path.splitext(file)
        file_dir=os.path.join(curDir,file)
        if filename!="kodim01":
            continue
        #print("filename==",filename)
        #print("file_dir==",file_dir)
        
        originImg=np.asarray(Image.open(file_dir).convert("RGB"))
        height,width,channel=originImg.shape
        for r in range(1,35):
            #print("r==",r)
            ImgQ_dir=os.path.join(save_dec_dir,filename + 'r{}.bmp'.format(r))
            #print("ImgQ_dir==",ImgQ_dir)
            JPEG2Kfile_dir=os.path.join(save_enc_dir,filename + 'r{}.j2k'.format(r))
            #print("JPEG2Kfile_dir==",JPEG2Kfile_dir)

            os.system('opj_compress -q {}'.format(r) + ' -i ' + file_dir + ' -o ' + JPEG2Kfile_dir)
            os.system('opj_decompress -i ' + JPEG2Kfile_dir + ' -o ' + ImgQ_dir)

            ImgQ=np.asarray(Image.open(ImgQ_dir).convert("RGB"))
            fsize = os.path.getsize(JPEG2Kfile_dir)
            bpp=fsize*8/height/width
            #print("bpp==",bpp)
            psnr=tf.image.psnr(originImg,ImgQ,max_val=255)
            #print(psnr)
            ssim=tf.image.ssim(originImg,ImgQ,max_val=255)
            #print(ssim)
            msssim=tf.image.ssim_multiscale(originImg,ImgQ,max_val=255)
            #print(msssim)
            J2KBPPs.append(bpp)
            J2KPSNRs.append(psnr.numpy())
            J2KSSIMs.append(ssim.numpy())
            J2KMSSSIMs.append(msssim.numpy())




# JPEG
# The currently supported image file formats are: 
#  PPM (PBMPLUS color format),
#  PGM (PBMPLUS grayscale format), 
#  BMP, 
#  Targa ，
#  RLE (Utah Raster Toolkit format)


JPEGBPPs=[]
JPEGPSNRs=[]
JPEGSSIMs=[]
JPEGMSSSIMs=[]
data_dir='H:\GradDesign\dataset\kodak_bmp'
os.chdir(jpeg_dir)
save_enc_dir = os.path.join(jpeg_dir,r'encImg_JPEG')   
save_dec_dir = os.path.join(jpeg_dir,r'decImg_JPEG')

if not os.path.exists(save_enc_dir):
    os.makedirs(save_enc_dir)
if not os.path.exists(save_dec_dir):
    os.makedirs(save_dec_dir)

for curDir,dirs,files in os.walk(data_dir):
    for file in files:
        filename,postfix=os.path.splitext(file)
        file_dir=os.path.join(curDir,file)
        if filename!="kodim01":
            continue
        #print("filename==",filename)
        #print("file_dir==",file_dir)
        originImg=np.asarray(Image.open(file_dir).convert("RGB"))
        height,width,channel=originImg.shape
        for q in range(1,40):
            #print("q==",q)
            ImgQ_dir=os.path.join(save_dec_dir,filename + 'q{}.bmp'.format(q))
            #print("ImgQ_dir==",ImgQ_dir)
            JPEGfile_dir=os.path.join(save_enc_dir,filename + 'q{}.jpg'.format(q))
            #print("JPEGfile_dir==",JPEGfile_dir)
            os.system('cjpeg -quality {}'.format(q) + ' ' + file_dir + ' ' + JPEGfile_dir)
            #print('cjpeg -quality {}'.format(q) + ' ' + file_dir + ' ' + JPEGfile_dir)
            os.system('djpeg -bmp ' + JPEGfile_dir + ' ' + ImgQ_dir)
            ImgQ=np.asarray(Image.open(JPEGfile_dir).convert("RGB"))
            fsize = os.path.getsize(JPEGfile_dir)
            bpp=fsize*8/height/width
            #print("bpp==",bpp)
            psnr=tf.image.psnr(originImg,ImgQ,max_val=255)
            #print(psnr)
            ssim=tf.image.ssim(originImg,ImgQ,max_val=255)
            #print(ssim)
            msssim=tf.image.ssim_multiscale(originImg,ImgQ,max_val=255)
            #print(msssim)
            JPEGBPPs.append(bpp)
            JPEGPSNRs.append(psnr.numpy())
            JPEGSSIMs.append(ssim.numpy())
            JPEGMSSSIMs.append(msssim.numpy())

print("JPEG")
print("JPEGBPPs=",JPEGBPPs)
print("JPEGPSNRs=",JPEGPSNRs)
print("JPEGSSIMs=",JPEGSSIMs)
print("JPEGMSSSIMs=",JPEGMSSSIMs)
print("J2K")
print("J2KBPPs=",J2KBPPs)
print("J2KPSNRs=",J2KPSNRs)
print("J2KSSIMs=",J2KSSIMs)
print("J2KMSSSIMs=",J2KMSSSIMs)
print("WebP")
print("WebPBPPs=",WebPBPPs)
print("WebPPSNRs=",WebPPSNRs)
print("WebPSSIMs=",WebPSSIMs)
print("WebPMSSSIMs=",WebPMSSSIMs)
print("BPG")
print("BPGBPPs=",BPGBPPs)
print("BPGPSNRs=",BPGPSNRs)
print("BPGSSIMs=",BPGSSIMs)
print("BPGMSSSIMs=",BPGMSSSIMs)