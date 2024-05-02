
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow import keras
#from keras import layer
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_addons.layers import InstanceNormalization
import time
import pdb

raw_image_dataset = tf.data.TFRecordDataset('k19.tfrecords')
# Create a dictionary describing the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
   # Parse the input tf.Example proto using the dictionary above.
    dataset=tf.io.parse_single_example(example_proto, image_feature_description)
    image_png=tf.io.decode_png(dataset['image'])
    image_png=tf.image.convert_image_dtype(image_png,dtype=tf.float32)
    image_png=tf.image.resize(image_png,[256,384])
    dataset['image']=2*image_png-1
    return dataset

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset=parsed_image_dataset.shuffle(8).batch(4)

BottleNeck=8

def quantizier_model(w,L=5,temperature=1):
    #print("quantizier is called")
    centers = tf.cast(tf.range(-2,3), tf.float32)
    w_stack = tf.stack([w for _ in range(L)], axis=-1)
    w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
    smx = tf.keras.activations.softmax(-1.0/temperature * tf.abs(w_stack - centers), axis=-1)
    w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)
    w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)
    #print("w_bar=",w_bar.shape)
    #print("quantizier return")
    return w_bar

def mypad(x,size):
    return tf.pad(x,[[0, 0], [size, size], [size, size], [0, 0]], 'REFLECT')

def encoderGC_model(C=BottleNeck):
    #print("encoderGC is called")
    input=layers.Input(shape=(256,384,3))
    #print("input_shape=",input.shape)
    x=layers.Lambda(mypad,arguments={'size':3})(input)

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
    x=layers.ReLU()(x)
    
    x=layers.Lambda(mypad,arguments={'size':1})(x)

    x=layers.Conv2D(C,kernel_size=(3,3),strides=(1,1),padding='valid')(x)
    x=InstanceNormalization()(x)
    output=layers.ReLU()(x)

    model=tf.keras.Model(inputs=input,outputs=output)

    return model
    
def generator_model(C=BottleNeck):
    #print("generator called")
    def residual_block(x, n_filters,kernel_size=3):
        #print("resblk called")
    # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        strides = [1,1]
        identity_map = x

        p = int((kernel_size-1)/2)
        res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = layers.Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='VALID')(res)
        res = InstanceNormalization()(res)
        res = layers.ReLU()(res)

        res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = layers.Conv2D(n_filters, kernel_size=(3,3), strides=(1,1),padding='VALID')(res)
        res = InstanceNormalization()(res)

        out = tf.add(res, identity_map)
        #print("resblk return")
        return out

    def upsample_block(n_filters,shape):
        #print("upsblk called")
        shape=[shape[1],shape[2],shape[3]]
        input=layers.Input(shape=shape)
        x=layers.Conv2DTranspose(n_filters,kernel_size=(3,3),strides=(2,2),padding='same')(input)
        x=InstanceNormalization()(x)
        output=layers.ReLU()(x)
        model=tf.keras.Model(inputs=[input],outputs=output)
        #print("upsblk return")
        return model

    input=layers.Input(shape=(16,24,C))
    
    x=layers.Lambda(mypad,arguments={'size':1})(input)

    x=layers.Conv2D(960,kernel_size=(3,3),strides=(1,1),padding='VALID')(x)
    x=InstanceNormalization()(x)
    res=layers.ReLU()(x)

    """
    res=residual_block(upsampled,960)
    res=residual_block(res,960)
    res=residual_block(res,960)
    res=residual_block(res,960)
    res=residual_block(res,960)
    res=residual_block(res,960)
    res=residual_block(res,960)
    res=residual_block(res,960)
    res=residual_block(res,960)

    ups=upsample_block(480,res.shape)(res)
    ups=upsample_block(240,ups.shape)(ups)
    ups=upsample_block(120,ups.shape)(ups)
    ups=upsample_block(60,ups.shape)(ups)
    """

    res1=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res1")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res1, res])

    res2=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res2")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res2, res])

    res3=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res3")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res3, res])

    res4=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res4")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    #pdb.set_trace()
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res4, res])

    res5=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res5")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res5, res])

    res6=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res6")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res6, res])

    res7=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res7")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res7, res])

    res8=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res8")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res8, res])

    res9=res
    res=layers.Lambda(mypad,arguments={'size':1},name="res9")(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res = layers.ReLU()(res)
    fac = layers.GlobalAveragePooling2D()(res)
    fac = layers.Dense(960//16, activation='relu')(fac)  # Assuming a matching size
    fac = layers.Dense(960, activation='sigmoid')(fac)
    fac = layers.Reshape((1, 1, 960))(fac) 
    res = layers.Multiply()([res, fac]) 
    res=layers.Lambda(mypad,arguments={'size':1})(res)
    res=layers.Conv2D(960, kernel_size=[3,3], strides=[1,1])(res)
    res=InstanceNormalization()(res)
    res=layers.Add()([res9, res])

    res=layers.Conv2DTranspose(480,kernel_size=[3,3],strides=[2,2],padding='same')(res)
    res=InstanceNormalization()(res)
    res=layers.ReLU()(res)

    res=layers.Conv2DTranspose(240,kernel_size=[3,3],strides=[2,2],padding='same')(res)
    res=InstanceNormalization()(res)
    res=layers.ReLU()(res)

    res=layers.Conv2DTranspose(120,kernel_size=[3,3],strides=[2,2],padding='same')(res)
    res=InstanceNormalization()(res)
    res=layers.ReLU()(res)

    res=layers.Conv2DTranspose(60,kernel_size=[3,3],strides=[2,2],padding='same')(res)
    res=InstanceNormalization()(res)
    res=layers.ReLU()(res)

    ups=layers.Lambda(mypad,arguments={'size':3})(res)
    
    ups=layers.Conv2D(3,kernel_size=(7,7),strides=(1,1),padding='valid')(ups)
    ups=InstanceNormalization()(ups)

    #output=layers.ReLU()(ups)

    output=tf.keras.activations.tanh(ups)

    model=tf.keras.Model(inputs=input,outputs=output)
    #print("generator return")
    return model

def conv_block(x, filters, kernel_size=(3,3), strides=2, padding='same'):
    in_kwargs = {'center':True, 'scale': True}
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = InstanceNormalization(**in_kwargs)(x)
    x = layers.ReLU()(x)

    return x

def multiscale_discriminator_model():
    #print("mtidisc is called")

    x=layers.Input(shape=(256,384,3))
    x2=layers.AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x4=layers.AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x2)
    
    def discriminator(x):
        #print("disc is called")
        c1=layers.Conv2D(64,kernel_size=(4,4),strides=(2,2),padding='same')(x)
        c1=layers.LeakyReLU()(c1)
        c2=conv_block(c1,filters=128,kernel_size=(4,4))
        c3=conv_block(c2,filters=256,kernel_size=(4,4))
        c4=conv_block(c3,filters=512,kernel_size=(4,4))
        output=layers.Conv2D(1,kernel_size=(4,4),strides=(1,1),padding='same')(c4)

        #print("disc return")
        return output

    disc=discriminator(x)
    disc2=discriminator(x2)
    disc4=discriminator(x4)

    model=tf.keras.Model(inputs=x,outputs=[disc,disc2,disc4])
    #print("mtidisc return")
    return model

encoderGC=encoderGC_model()
quantizier=quantizier_model
generator_optimizer=tf.keras.optimizers.Adam(0.0002)
disciminator_optimizer=tf.keras.optimizers.Adam(0.0002)
generator=generator_model()
discriminator=multiscale_discriminator_model()

def generator_loss(fake_loss):
    #print("genloss is called")
    G_loss=tf.reduce_mean(tf.square(fake_loss-1.))
    #print("genloss return")
    return G_loss

def discriminator_loss(real_loss,real_loss2,real_loss4,fake_loss,fake_loss2,fake_loss4):
    #print("discloss is called")
    D_loss=tf.reduce_mean(tf.square(real_loss-1.))+tf.reduce_mean(tf.square(real_loss2-1.))+tf.reduce_mean(tf.square(real_loss4-1.))+tf.reduce_mean(tf.square(fake_loss-1.))+tf.reduce_mean(tf.square(fake_loss2-1.))+tf.reduce_mean(tf.square(fake_loss4-1.))
    #print("discloss return")
    return D_loss

def train_step(origin_image):
    #print("train_step is called")
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        w=encoderGC(origin_image)
        #print(type(w))
        #print(w.shape)
        w_hat=quantizier(w)
        #print(type(w_hat))
        #print(w_hat.shape)
        generated_images=generator(w_hat,training=True)
        import pdb
        #pdb.set_trace()
        real_output,real2_output,real4_output=discriminator(origin_image,training=True)
        fake_output,fake2_output,fake4_output=discriminator(generated_images,training=True)

        disc_loss=discriminator_loss(real_output,real2_output,real4_output,fake_output,fake2_output,fake4_output)
        gen_loss=generator_loss(fake_output)+12 * tf.keras.losses.MeanSquaredError()(origin_image, generated_images)

    gradient_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)
    gradient_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_of_generator,generator.trainable_variables))
    disciminator_optimizer.apply_gradients(zip(gradient_of_discriminator,discriminator.trainable_variables))
    #print("train_step return")

EPOCH=1026*3



def save_img(model,image,epoch,index):
    gen_imgs=model(image,training=False)
    for i in range(gen_imgs.shape[0]):
        image=(gen_imgs[i,:,:,:]+1)/2
        image=tf.math.multiply(image,255)
        image=tf.cast(image,dtype=tf.uint8)
        image=tf.image.encode_png(image)
        with tf.io.gfile.GFile('../Semcom/GAN/chztestk19/C{:02d}_image_{:02d}_at_epoch_{:04d}_kodak.png'.format(BottleNeck,index,epoch), 'wb') as file:
            file.write(image.numpy())
        print('image_at_epoch_{:04d} have saved'.format(epoch))


def train(dataset,epochs):
    print("train begin")
    for epoch in range(epochs):
        count=0
        for example in dataset:
            #print("epoch=",epoch+1)
            image=example["image"].numpy()
            train_step(image)
            count+=1
            if epoch%25==0:
                w=encoderGC(example["image"])
                w_hat=quantizier(w)
                save_img(generator,w_hat,epoch+1,count)
    print("train finished")

train(parsed_image_dataset,EPOCH)

def quantizier_aftertrain(w,L=5):
    #print("quantizier is called")
    centers = tf.cast(tf.range(-2,3), tf.float32)
    w_stack = tf.stack([w for _ in range(L)], axis=-1)
    w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
    return w_hard

model_dir='../Semcom/GAN/chztestk19'

input_img=layers.Input(shape=(256,384,3))
x=input_img
for i in range(len(encoderGC.layers)):
    if not isinstance(encoderGC.layers[i],layers.InputLayer):
      x=encoderGC.layers[i](x)
w_hat=layers.Lambda(quantizier_aftertrain,output_shape=(16, 24, 8))(x)
GC_encoder=tf.keras.Model(inputs=input_img,outputs=w_hat,name="encoder_quant")

import os

encoderGC.save(os.path.join(model_dir, 'GC_encoder_noQuant_keras_test'))
encoderGC.summary()
GC_encoder.save(os.path.join(model_dir, 'GC_encoder_Quant_keras_test'))
GC_encoder.summary()
generator.save(os.path.join(model_dir, 'GC_generator_keras_test'))
generator.summary()

input_img=layers.Input(shape=(256,384,3))
x=input_img
for i in range(len(GC_encoder.layers)):
    if not isinstance(GC_encoder.layers[i],layers.InputLayer):
        x=GC_encoder.layers[i](x)
#x=layers.Lambda(quantizier_aftertrain,output_shape=(32, 48, 8))(x)
for i in range(len(generator.layers)):
    if not isinstance(generator.layers[i],layers.InputLayer):
        temp=x
        if "res" in generator.layers[i].name:
            temp=x
            x=generator.layers[i](x)
        elif isinstance(generator.layers[i],layers.Add):
            x=generator.layers[i]([x,temp])
        elif isinstance(generator.layers[i],layers.Multiply):
            x=generator.layers[i]([x,temp])
        else:
            x=generator.layers[i](x)
            
output_img=x
GCModel=tf.keras.Model(inputs=input_img,outputs=output_img,name="GCModel")
GCModel.save(os.path.join(model_dir, 'GCModel_keras_test'))
GCModel.summary()



