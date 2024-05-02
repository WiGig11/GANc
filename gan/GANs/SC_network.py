import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization


Level=5
_BOTTLENECK=8
_EPOCH=1
one_hot=True
use_instance_map=True
use_FeatureLoss=True
use_VGGLoss=True
train_with_mask=True

_HEIGHT=128
_WIDTH=256
_CHANNEL_IMG=3

if one_hot:
    _CHANNEL_LABEL=35
else:
    _CHANNEL_LABEL=1

if use_instance_map:
    _CHANNEL_INST=1
else:
    _CHANNEL_INST=0

_NUM_D=2



raw_image_dataset = tf.data.TFRecordDataset('dataset/city/city_label_inst.tfrecords')
# Create a dictionary describing the features.
image_feature_description = {
    'image_left': tf.io.FixedLenFeature([], tf.string),
    'segmentation_label': tf.io.FixedLenFeature([], tf.string),
    'segmentation_instance': tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):

    def get_edges(inst_map):
        inst_map=tf.squeeze(inst_map,axis=2)
        edge_size=tf.cast(tf.shape(inst_map),dtype=tf.int64)
        paddings = tf.constant([[1, 1,], [1, 1]])
        inst_edge_pad=tf.pad(inst_map, paddings, "SYMMETRIC")
        edge=tf.zeros_like(inst_map,dtype=tf.bool)
        edgeIdx = tf.where(edge[:,:]
             | (inst_edge_pad[1:-1,1:-1] != inst_edge_pad[1:-1,2:])
             | (inst_edge_pad[1:-1,1:-1] != inst_edge_pad[1:-1,2:])
             | (inst_edge_pad[1:-1,1:-1] != inst_edge_pad[2:,1:-1])
             | (inst_edge_pad[1:-1,1:-1] != inst_edge_pad[2:,1:-1])
            )
        edge_map = tf.scatter_nd(edgeIdx, tf.ones(tf.shape(edgeIdx)[0],dtype=tf.int64), edge_size)
        edge_map=tf.cast(edge_map,dtype=tf.uint8)
        
        edge_map=tf.expand_dims(edge_map,axis=2)
    
        return edge_map
   # Parse the input tf.Example proto using the dictionary above.
    dataset=tf.io.parse_single_example(example_proto, image_feature_description)
    image_png=tf.io.decode_png(dataset['image_left'])
    map_png=tf.io.decode_png(dataset['segmentation_label'])
    inst_png=tf.io.decode_png(dataset['segmentation_instance'],dtype=tf.uint16)

    img=tf.image.convert_image_dtype(image_png,dtype=tf.float32)
    img=tf.image.resize(img,[_HEIGHT,_WIDTH])
    dataset["image_left"]=2*img-1
    
    if one_hot:
        labelMap=tf.image.resize(map_png,[_HEIGHT,_WIDTH])
        labelMap=tf.expand_dims(labelMap, axis=0)
        size = tf.shape(labelMap)
        oneHot_size = (size[0], size[1], size[2], 35) # num of labels is 35 in cityscapes
        labelIdx = tf.where(labelMap >= 0)  # shape: [b, h, w, c] -> [b*h*w*c, 4]
        idx = tf.concat([labelIdx[..., :-1], tf.cast(tf.reshape(labelMap, [-1, 1]), tf.int64)], 1)
        input_label = tf.scatter_nd(idx, tf.ones(tf.shape(idx)[0]), oneHot_size)
        input_label=tf.image.convert_image_dtype(input_label,dtype=tf.float32)
        input_label=tf.image.resize(input_label,[_HEIGHT,_WIDTH])
        input_label=tf.squeeze(input_label,axis=0)
        dataset["segmentation_label"]=2*input_label-1
    else:
        input_label=tf.image.convert_image_dtype(map_png,dtype=tf.float32)
        input_label=tf.image.resize(input_label,[_HEIGHT,_WIDTH])
        dataset["segmentation_label"]=2*input_label-1

    
    inst_edge_png=get_edges(inst_png)
    inst_edge_png=tf.image.convert_image_dtype(inst_edge_png,dtype=tf.float32)
    inst_edge_png=tf.image.resize(inst_edge_png,[_HEIGHT,_WIDTH])
    dataset["segmentation_instance"]=2*inst_edge_png-1

    return dataset

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset=parsed_image_dataset.take(1)
parsed_image_dataset=parsed_image_dataset.shuffle(1).batch(1)





# quantizer
class NetQuantizer(layers.Layer):
    def __init__(self,L=Level,temperature=1):
        super(NetQuantizer,self).__init__()
        #self.centers=tf.cast(range(int(-(L-1)/2),int((L-1)/2)),tf.float32)
        self.centers=tf.cast(range(-2,3),tf.float32)
        self.L=L
        self.temperature=temperature

    def bulid(self,input_shape):
        pass

    def call(self,input):
        w=input
        w_stack=tf.stack([w for _ in range(self.L)],axis=-1)
        w_hard=tf.cast(tf.argmin(tf.math.abs(w_stack-self.centers),axis=-1),tf.float32)+tf.math.reduce_min(self.centers)
        smx=tf.keras.activations.softmax(-1.0/self.temperature * tf.math.abs(w_stack - self.centers),axis=-1)
        w_soft = tf.einsum('ijklm,m->ijkl', smx, self.centers)  
        w_hat = tf.stop_gradient(w_hard - w_soft) + w_soft
        return w_hat


class ReflectionPad2d(layers.Layer):
    def __init__(self, paddings=(1, 1), **kwargs):
        self.paddings = tuple(paddings)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPad2d, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1]+self.paddings[0], s[2]+self.paddings[1], s[3])

    def call(self, x):
        w_pad, h_pad = self.paddings
        x = tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')
        return x


class NetEncoderSC(tf.keras.Model):
    """
    Image encoder: c7s1-60, d120, d240, d480, c3s1-C, q, c3s1-480, d960
    """
    def __init__(self,bottleneckDepth=_BOTTLENECK):
        super(NetEncoderSC,self).__init__(name='NetEncoderSC')
        self.bottleneckDepth=bottleneckDepth

        input_img=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_CHANNEL_IMG))
        input_smtMap=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_CHANNEL_LABEL+_CHANNEL_INST))
        input=layers.Concatenate(axis=-1)([input_img,input_smtMap])
        
        # padding
        x=ReflectionPad2d(paddings=(3,3))(input)
        # c7s1-60
        x=layers.Conv2D(60,kernel_size=(7,7),strides=(1,1),padding='valid')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        # d120
        x=layers.Conv2D(120,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        # d240
        x=layers.Conv2D(240,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        # d480
        x=layers.Conv2D(480,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        # padding
        x=ReflectionPad2d(paddings=(1,1))(x)
        # c3s1-C
        x=layers.Conv2D(bottleneckDepth,kernel_size=(3,3),strides=(1,1),padding='valid')(x)
        x=InstanceNormalization()(x)
        output_feature=layers.ReLU()(x)

        self.model=tf.keras.Model(inputs=[input_img,input_smtMap],outputs=output_feature)

    def bulid(self,input_shape):
        pass

    def call(self,input_image,input_semanticMap):
        output_featureMap=self.model((input_image,input_semanticMap))
        return output_featureMap

class NetFeatureExtractor(tf.keras.Model):
    """
    Semantic label map encoder: c7s1-60, d120, d240, d480, d960
    Feature extractor in the paper
    """
    def __init__(self):
        super(NetFeatureExtractor,self).__init__(name='NetFeatureExtractor')
        
        input_smtMap=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_CHANNEL_LABEL+_CHANNEL_INST))
        
        x=ReflectionPad2d(paddings=(3,3))(input_smtMap)
        
        x=layers.Conv2D(60,kernel_size=(7,7),strides=(1,1),padding='valid')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        
        x=layers.Conv2D(120,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        
        x=layers.Conv2D(240,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        
        x=layers.Conv2D(480,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        
        x=layers.Conv2D(960,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        output_feature=layers.ReLU()(x)
        
        self.model=tf.keras.Model(inputs=input_smtMap,outputs=output_feature)

    def bulid(self,input_shape):
        pass

    def call(self,input_semanticMap):
        output_featureMap=self.model(input_semanticMap)
        return output_featureMap


class ResnetBlock(layers.Layer):
    def __init__(self, filters,use_dropout=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        # CONSTANT REFLECT SYMMETRIC
        model = tf.keras.Sequential()
        
        model.add(layers.ZeroPadding2D(padding=(1, 1)))
        model.add(layers.Conv2D(filters, kernel_size=[3,3], strides=[1,1]))
        model.add(InstanceNormalization())
        model.add(layers.ReLU())

        if use_dropout:
            model.add(layers.Dropout(0.5))
        
        model.add(layers.ZeroPadding2D([1, 1]))
        model.add(layers.Conv2D(filters, kernel_size=[3,3], strides=[1,1]))
        model.add(InstanceNormalization())

        self.model = model

    def call(self, input):
        identity = input
        x = self.model(input)
        output = x + identity
        return output

class UpSampleBlock(layers.Layer):
    def __init__(self, filters,**kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        # CONSTANT REFLECT SYMMETRIC
        model = tf.keras.Sequential()       
        model.add(layers.Conv2DTranspose(filters,kernel_size=[3,3],strides=[2,2],padding='same'))
        model.add(InstanceNormalization())
        model.add(layers.ReLU())
        self.model = model

    def call(self, input):
        output=self.model(input)
        return output

# Mask
class BinaryMask(layers.Layer):
    def __init__(self):
        super(BinaryMask,self).__init__()
        self.numMask=0

    def getMask(self,oriMask,height,width):
        zero_idx=np.argwhere(np.all(oriMask==-1,axis=(0,1,2)))
        newMask=np.delete(oriMask,zero_idx,axis=3)
        #newMask=tf.concat([newMask,tf.zeros(shape=(height,width))],axis=-1)
        self.numMask=newMask.shape[3]
        return newMask

    def call(self,input_w,input_mask):
        _,height,width,channel=input_w.shape
        maskList=self.getMask(input_mask[:,:,:,:35],height,width)
        maskList=tf.image.resize(maskList,size=[height,width])
        outputs=tf.zeros(shape=(1,height,width,channel))
        for maskIdx in range(self.numMask):
            singleoutput=tf.math.multiply(input_w,maskList[:,:,:,maskIdx:maskIdx+1])
            outputs=tf.concat([outputs,singleoutput],axis=0)
        return outputs


class NetGeneratorSC(tf.keras.Model):
    """
    Generator/decoder: c3s1-960, R960, R960, R960, R960, R960, R960, R960, R960, R960, u480, u240, u120, u60, c7s1-3
    """
    def __init__(self):
        super(NetGeneratorSC,self).__init__(name='NetGeneratorSC')
        
        input_feaImgMap=tf.keras.Input(shape=(_HEIGHT//8,_WIDTH//8,_BOTTLENECK))
        input_feaSmtMap=tf.keras.Input(shape=(_HEIGHT//16,_WIDTH//16,960))
        
        # padding
        x=ReflectionPad2d(paddings=(1,1))(input_feaImgMap)
        # c3s1-480
        x=layers.Conv2D(480,kernel_size=(3,3),strides=(1,1),padding='valid')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)
        # d960
        x=layers.Conv2D(960,kernel_size=(3,3),strides=(2,2),padding='same')(x)
        x=InstanceNormalization()(x)
        x=layers.ReLU()(x)

        input=layers.Concatenate(axis=-1)([x,input_feaSmtMap])

        # padding
        y=ReflectionPad2d(paddings=(1,1))(input)
        # c3s1-960
        y=layers.Conv2D(960,kernel_size=(3,3),strides=(1,1),padding='valid')(y)
        y=InstanceNormalization()(y)
        y=layers.ReLU()(y)

        # R960 * 1
        y=ResnetBlock(filters=960)(y)
        # R960 * 2
        y=ResnetBlock(filters=960)(y)
        # R960 * 3
        y=ResnetBlock(filters=960)(y)
        # R960 * 4
        y=ResnetBlock(filters=960)(y)
        # R960 * 5
        y=ResnetBlock(filters=960)(y)
        # R960 * 6
        y=ResnetBlock(filters=960)(y)
        # R960 * 7
        y=ResnetBlock(filters=960)(y)
        # R960 * 8
        y=ResnetBlock(filters=960)(y)
        # R960 * 9
        y=ResnetBlock(filters=960)(y)
        # u480
        y=UpSampleBlock(filters=480)(y)
        # u240
        y=UpSampleBlock(filters=240)(y)
        # u120
        y=UpSampleBlock(filters=120)(y)
        # u60
        y=UpSampleBlock(filters=60)(y)
        #padding
        y=ReflectionPad2d(paddings=(3,3))(y)
        # c7s1-3
        y=layers.Conv2D(3,kernel_size=(7,7),strides=(1,1),padding='valid')(y)
        y=InstanceNormalization()(y)

        output_genImg=layers.ReLU()(y)
        # model.add(tf.keras.activations.tanh())

        self.model=tf.keras.Model(inputs=[input_feaImgMap,input_feaSmtMap],outputs=output_genImg)


    def bulid(self,input_shape):
        pass

    def call(self,input_featureImage,input_featureSemanticMap):
        output_generateImage=self.model((input_featureImage,input_featureSemanticMap))
        return output_generateImage



# Discriminator
class NetMultiscaleDiscriminator(tf.keras.Model):
    def __init__(self, num_D=3, n_layers=3, getIntermFeat=False):
        super(NetMultiscaleDiscriminator, self).__init__(name='NetMultiscaleDiscriminator')
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD)

        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def singleD_forward(self, model, x):
        if self.getIntermFeat:
            result = [x]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(x)]

    def call(self, x):
        num_D = self.num_D
        result = []
        input_downsampled = x
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(tf.keras.Model):
    def __init__(self, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__(name='NLayerDiscriminator')
        self.getIntermFeat = getIntermFeat
        

        sequence = tf.keras.Sequential()
        sequence.add(layers.InputLayer([None, None, _CHANNEL_IMG+_CHANNEL_LABEL+_CHANNEL_INST]))
        sequence.add(layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same'))
        sequence.add(layers.LeakyReLU(0.2))

        
        sequence.add(layers.Conv2D(128, kernel_size=(4,4), strides=(2,2), padding='same'))
        sequence.add(InstanceNormalization())
        sequence.add(layers.LeakyReLU(0.2))

        sequence.add(layers.Conv2D(256, kernel_size=(4,4), strides=(2,2), padding='same'))
        sequence.add(InstanceNormalization())
        sequence.add(layers.LeakyReLU(0.2))

        
        sequence.add(layers.Conv2D(512, kernel_size=(4,4), strides=(1,1)))
        sequence.add(layers.ZeroPadding2D())
        sequence.add(InstanceNormalization())
        sequence.add(layers.LeakyReLU(0.2))

        sequence.add(layers.Conv2D(1, kernel_size=(4,4), strides=(1,1)))
        sequence.add(layers.ZeroPadding2D())

        if getIntermFeat:
            for n in range(len(sequence.layers)):
                setattr(self, 'model'+str(n), sequence.layers[n])
        else:
            self.model = sequence

    def call(self, x):
        if self.getIntermFeat:
            res = [x]
            for n in range(3+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(x)

from tensorflow.keras.applications.vgg19 import VGG19

class Vgg19(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Vgg19, self).__init__(name='Vgg19')
        vgg_pretrained_features = VGG19(weights='imagenet', include_top=False)
        if trainable is False:
            vgg_pretrained_features.trainable = False
        vgg_pretrained_features = vgg_pretrained_features.layers
        self.slice1 = tf.keras.Sequential()
        self.slice2 = tf.keras.Sequential()
        self.slice3 = tf.keras.Sequential()
        self.slice4 = tf.keras.Sequential()
        self.slice5 = tf.keras.Sequential()
        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 13):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(13, 18):
            self.slice5.add(vgg_pretrained_features[x])

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(layers.Layer):
    def __init__(self):
        super(VGGLoss, self).__init__(name='VGGLoss')
        self.vgg = Vgg19()
        self.criterion = tf.keras.losses.MeanAbsoluteError()  # lambda ta, tb: tf.reduce_mean(tf.abs(ta - tb))
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            y_vgg_temp = tf.stop_gradient(y_vgg[i])
            loss += self.layer_weights[i] * self.criterion(x_vgg[i], y_vgg_temp)
        return loss

encoder=NetEncoderSC()
extractor=NetFeatureExtractor()
quantizier=NetQuantizer()
maskapplier=BinaryMask()
generator=NetGeneratorSC()
discriminator=NetMultiscaleDiscriminator(num_D=_NUM_D)

# LSGAN
criterionGAN=tf.keras.losses.MeanSquaredError()

if use_FeatureLoss:
    criterionFeat = tf.keras.losses.MeanAbsoluteError()
if use_VGGLoss:
    criterionVGG = VGGLoss()

def train_G_D(real_img, input_label):
    
    w = encoder(real_img,input_label)
    featureMap=extractor(input_label)
    w_hat=quantizier(w)
    
    loss_D_fake=0
    loss_D_real=0
    loss_G_GAN=0
    loss_G_GAN_Feat = 0
    loss_G_VGG = 0

    lambda_feat=10
    
    fake_img=generator(w_hat,featureMap)
    
    real_pair = tf.concat([real_img,input_label], axis=-1)
    fake_pair = tf.concat([fake_img,input_label], axis=-1)

    # Fake Detection and Loss
    pred_fake_pool = discriminator(fake_pair)
    loss_D_fake += discriminate(pred_fake_pool, False) 

    # Real Detection and Loss
    pred_real = discriminator(real_pair)
    loss_D_real += discriminate(pred_real, True) 

    
    # GAN loss (Fake Passability Loss)
    #pred_fake = discriminator(fake_pair)
    #loss_G_GAN = discriminate(pred_fake, False)
    

    # GAN loss (Fake Passability Loss)
    pred_fake = discriminator(fake_pair)
    loss_G_GAN += discriminate(pred_fake, True) 
    
    if use_FeatureLoss:
        feat_weights = 4.0 / (3 + 1)
        D_weights = 1.0 / _NUM_D
        for i in range(_NUM_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                        criterionFeat(pred_fake[i][j], pred_real[i][j]) * lambda_feat
        if use_VGGLoss:
            loss_G_VGG += criterionVGG(fake_img, real_img) * lambda_feat 

    return loss_D_fake, loss_D_real, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, fake_img


def train_G_D_with_mask(real_img, input_label):
    
    w = encoder(real_img,input_label)
    featureMap=extractor(input_label)
    w_hat=quantizier(w)
    
    w_hat_mask=maskapplier(w_hat,input_label)

    
    loss_D_fake=0
    loss_D_real=0
    loss_G_GAN=0
    loss_G_GAN_Feat = 0
    loss_G_VGG = 0

    lambda_feat=10
    fake_imgs=[]

    for idx in range(maskapplier.numMask):
        fake_img=generator(w_hat_mask[idx:idx+1,:,:,:],featureMap)
        fake_imgs.append(fake_img)
        real_pair = tf.concat([real_img,input_label], axis=-1)
        fake_pair = tf.concat([fake_img,input_label], axis=-1)

        # Fake Detection and Loss
        pred_fake_pool = discriminator(fake_pair)
        loss_D_fake += discriminate(pred_fake_pool, False) / maskapplier.numMask

        # Real Detection and Loss
        pred_real = discriminator(real_pair)
        loss_D_real += discriminate(pred_real, True) / maskapplier.numMask

        """
        # GAN loss (Fake Passability Loss)
        pred_fake = discriminator(fake_pair)
        loss_G_GAN = discriminate(pred_fake, False)
        """

        # GAN loss (Fake Passability Loss)
        pred_fake = discriminator(fake_pair)
        loss_G_GAN += discriminate(pred_fake, True) / maskapplier.numMask
        loss_G_GAN_Feat_single=0
        if use_FeatureLoss:
            feat_weights = 4.0 / (3 + 1)
            D_weights = 1.0 / _NUM_D
            for i in range(_NUM_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat_single += D_weights * feat_weights * \
                        criterionFeat(pred_fake[i][j], pred_real[i][j]) * lambda_feat
        loss_G_GAN_Feat += loss_G_GAN_Feat_single / maskapplier.numMask
        if use_VGGLoss:
            loss_G_VGG += criterionVGG(fake_img, real_img) * lambda_feat / maskapplier.numMask

    return loss_D_fake, loss_D_real, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, fake_imgs







def discriminate(input, target_is_real):
    if isinstance(input[0], list):
        loss = 0
        for input_i in input:
            pred = input_i[-1]
            if target_is_real:
                target_tensor = tf.ones_like(pred)
            else:
                target_tensor = tf.zeros_like(pred)
            loss += criterionGAN(pred, target_tensor)
        return loss
    else:
        if target_is_real:
            target_tensor = tf.ones_like(input[-1])
        else:
            target_tensor = tf.zeros_like(input[-1])
        return criterionGAN(input[-1], target_tensor)


optimizer_G=tf.keras.optimizers.Adam(0.0002)
optimizer_D=tf.keras.optimizers.Adam(0.0002)
optimizer_E=tf.keras.optimizers.Adam(0.0002)
optimizer_F=tf.keras.optimizers.Adam(0.0002)


def train_step(real_img,input_label,train_with_mask=False):
    # input_label = inputs[0]
    with tf.GradientTape() as gen_ext_enc_tape, tf.GradientTape() as disc_tape,tf.GradientTape() as enc_tape,tf.GradientTape() as ext_tape:
        if train_with_mask:
            loss_D_fake, loss_D_real, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, fake_imgs = train_G_D_with_mask(real_img,input_label)
        else:
            loss_D_fake, loss_D_real, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, fake_imgs = train_G_D(real_img,input_label)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

    gen_ext_enc_grads = gen_ext_enc_tape.gradient(loss_G, [generator.trainable_variables,extractor.trainable_variables,encoder.trainable_variables])
    disc_grads = disc_tape.gradient(loss_D, discriminator.trainable_variables)
    """
    enc_grads = gen_grads.gradient(loss_G, encoder.trainable_variables)
    ext_grads = gen_grads.gradient(loss_G, extractor.trainable_variables)
    optimizer_E.apply_gradients(zip(enc_grads,
                                    encoder.trainable_variables))
    optimizer_F.apply_gradients(zip(ext_grads,
                                    extractor.trainable_variables))
    """
    optimizer_G.apply_gradients(zip(gen_ext_enc_grads[0],
                                    generator.trainable_variables))  
    optimizer_F.apply_gradients(zip(gen_ext_enc_grads[1],
                                    extractor.trainable_variables))                              
    optimizer_E.apply_gradients(zip(gen_ext_enc_grads[2],
                                    encoder.trainable_variables))  
    optimizer_D.apply_gradients(zip(disc_grads,
                                    discriminator.trainable_variables))

    loss_D_dict = {
        'D_fake': loss_D_fake,
        'D_real': loss_D_real
        }
    loss_G_dict = {
        'G_GAN': loss_G_GAN,
        'G_GAN_Feat': loss_G_GAN_Feat,
        'loss_G_VGG': loss_G_VGG
    }
    return loss_D_dict, loss_G_dict, fake_imgs

def save_img(image,epoch,index,mask_index=36):
    image=(image+1)/2
    image=tf.math.multiply(image,255)
    image=tf.cast(image,dtype=tf.uint8)
    image=tf.image.encode_png(image)
    with tf.io.gfile.GFile('./SemCom/GAN/img_SC_pc/C{:02d}_image_{:02d}_mask_{:02d}_at_epoch_{:04d}.png'.format(_BOTTLENECK,index,mask_index,epoch), 'wb') as file:
        file.write(image.numpy())
    #print('image_at_epoch_{:04d} have saved'.format(epoch))

import time 
def train(dataset,epoches):
    for epoch in range(epoches):
        start=time.time()
        for index,example in enumerate(dataset):
            image=example["image_left"]
            semanticLabel=example["segmentation_label"]
            semanticInst=example["segmentation_instance"]
            if use_instance_map:
                semanticMap=tf.concat([semanticLabel,semanticInst],axis=-1)
            else:
                semanticMap=semanticLabel
            _,_,fake_images=train_step(image,semanticMap,train_with_mask)
            if epoch % 20==0:
                if train_with_mask:    
                    for mask_index,fake_image in enumerate(fake_images):
                        fake_image=tf.squeeze(fake_image,axis=0)
                        save_img(fake_image,epoch,index,mask_index)
                else:
                    fake_image=tf.squeeze(fake_images,axis=0)
                    save_img(fake_image,epoch,index)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(parsed_image_dataset,_EPOCH)
print("OK")


def test(dataset):
    for index,example in enumerate(dataset):
        image=example["image_left"]
        semanticLabel=example["segmentation_label"]
        semanticInst=example["segmentation_instance"]
        if use_instance_map:
            semanticMap=tf.concat([semanticLabel,semanticInst],axis=-1)
        else:
            semanticMap=semanticLabel

        w = encoder(image,semanticMap)
        featureMap=extractor(semanticMap)
        w_hat=quantizier(w)
        w_hat_mask=maskapplier(w_hat,semanticMap)

        for idx_mask in range(maskapplier.numMask):
            fake_img=generator(w_hat_mask[idx_mask:idx_mask+1,:,:,:],featureMap)
            fake_img=tf.squeeze(fake_img,axis=0)
            save_img(fake_img,9999,index,idx_mask)
            
test(parsed_image_dataset)