
from pickletools import optimize
from re import X
from importlib_metadata import re
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

class args():
    dataset_train="kodak"
    data_dir_train="./dataset/kodak"
    batch_size_train=1
    repeatNum=10

    model_dir="./JSCC/saveModel"
    
    train_epoch=1
    train_channel_snr=20
    conv_depth=16

    channel="awgn"
    learning_rate=0.0001

def get_dataset(is_training, data_dir):
    """Returns a dataset object"""
    #maybe_download_and_extract(data_dir)
    file_pattern = os.path.join(data_dir, "kodim*.png")
    file_pattern = file_pattern.replace('\\', '/')
    filename_dataset = tf.data.Dataset.list_files(file_pattern)
    return filename_dataset.map(lambda x: tf.image.decode_png(tf.io.read_file(x)))

_HEIGHT = 512
_WIDTH = 768
_NUM_CHANNELS = 3

def parse_record(raw_record, dtype):
    """Parse kodak image and label from a raw record."""
    image = tf.reshape(raw_record, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
    # normalise images to range 0-1
    image = tf.cast(image, dtype)
    image = tf.divide(image, 255.0)
    return image

class Encoder(tf.keras.Model):
    """Build encoder from specified arch"""

    def __init__(self, c, name="Encoder"):
        super(Encoder,self).__init__(name=name)
        self.c=c
        input_img=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_NUM_CHANNELS))
        mid=layers.Conv2D(16,kernel_size=5,strides=(2,2),padding='same')(input_img)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2D(32,kernel_size=5,strides=(2,2),padding='same')(mid)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2D(32,kernel_size=5,strides=(1,1),padding='same')(mid)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2D(32,kernel_size=5,strides=(1,1),padding='same')(mid)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2D(c,kernel_size=5,strides=(1,1),padding='same')(mid)
        output=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
 
        self.model=tf.keras.Model(inputs=input_img,outputs=output,name="JSCCEncoder")

    def call(self, x):
        y=self.model(x)
        return y

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.raw_ops.RandomStandardNormal(shape=tf.shape(x),  dtype=tf.float32)
    y = x + awgn
    return y

def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Rayleigh fading+AWGN
    Rician fading=sqrt(k/(k+1))+sqrt(1/(k+1))*Rayleigh fading
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    
    # channel gain
    if h is None:
        h = tf.complex(
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
        )

    # additive white gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )
    
    
    return x, h

def phase_invariant_fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.

    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    """
    # channel gain
    if h is None:
        n1 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
        n2 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)

        h = tf.sqrt(tf.square(n1) + tf.square(n2))

    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev / np.sqrt(2), dtype=tf.float32)
    """
    return x,h

class Channel(layers.Layer):
    def __init__(self, channel_type, channel_snr, name="Channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def call(self, inputs):
        prev_h=None
        encoded_img= inputs
        inter_shape = tf.shape(encoded_img)
        # reshape array to [-1, dim_z]
        z = layers.Flatten()(encoded_img)
        # convert from snr to std
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            dim_z = tf.shape(z)[1]
            # normalize latent vector so that the average power is 1
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out = real_awgn(z_in, noise_stddev)
            h = tf.ones_like(z_in)  # h just makes sense on fading channels

        elif self.channel_type == "fading":
            dim_z = tf.shape(z)[1] // 2
            # convert z to complex representation
            z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
            # normalize the latent vector so that the average power is 1
            z_norm = tf.reduce_sum(
                tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
            )
            z_in = z_in * tf.complex(
                tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
            )
            z_out, h = fading(z_in, noise_stddev, prev_h)
            # convert back to real
            z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)

        elif self.channel_type == "fading-real":
            # half of the channels are I component and half Q
            dim_z = tf.shape(z)[1] // 2
            # normalization
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)

        else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
        z_out = tf.reshape(z_out, inter_shape)
        # compute average power
        avg_power = tf.reduce_mean(tf.math.real(z_in * tf.math.conj(z_in)))
        # add avg_power as layer's metric
        return z_out, avg_power, h

class Decoder(tf.keras.Model):
    """Build encoder from specified arch"""

    def __init__(self, n_channels, name="Decoder"):
        super(Decoder,self).__init__(name=name)
        self.n_channels=n_channels

        input=tf.keras.Input(shape=(_HEIGHT//4,_WIDTH//4,args.conv_depth))
        mid=layers.Conv2DTranspose(32,kernel_size=5,strides=(1,1),padding='same')(input)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2DTranspose(32,kernel_size=5,strides=(1,1),padding='same')(mid)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2DTranspose(32,kernel_size=5,strides=(1,1),padding='same')(mid)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        mid=layers.Conv2DTranspose(16,kernel_size=5,strides=(2,2),padding='same')(mid)
        mid=layers.PReLU(alpha_initializer=tf.initializers.constant(0.1))(mid)
        output=layers.Conv2DTranspose(n_channels,kernel_size=5,strides=(2,2),padding='same',activation='sigmoid')(mid)

        self.model=tf.keras.Model(inputs=input,outputs=output,name="JSCCDecoder")


    def call(self, x):
        y=self.model(x)
        return y

class DeepJSCC(tf.keras.Model):
    def __init__(
        self,
        channel_snr,
        conv_depth,
        channel_type,
        name="deep_jscc",
        **kwargs
    ):
        super(DeepJSCC, self).__init__(name=name, **kwargs)

        n_channels = 3  # change this if working with BW images(i.e. Binary Image)
        self.encoder = Encoder(conv_depth)
        self.decoder = Decoder(n_channels)
        self.channel = Channel(channel_type, channel_snr)

        input_img=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_NUM_CHANNELS))
        mid=self.encoder(input_img)
        mid, avg_power, chn_gain=self.channel(mid)
        output_img=self.decoder(mid)
        self.model=tf.keras.Model(inputs=input_img,outputs=output_img,name="JSCCModel")
        self.testmodel1=self.get_cprs_model1()
        self.testmodel2=self.get_cprs_model2()

    def call(self, inputs):
        decoded_img = self.model(inputs)
        return decoded_img

    def change_channel_snr(self, channel_snr):
        self.channel.channel_snr = channel_snr

    def get_cprs_model1(self):
        input=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_NUM_CHANNELS))
        x=input
        for i in range(len(self.encoder.model.layers)):
            if not isinstance(self.encoder.model.layers[i],layers.InputLayer):
                x=self.encoder.model.layers[i](x)
        for i in range(len(self.decoder.model.layers)):
            if not isinstance(self.decoder.model.layers[i],layers.InputLayer):
                x=self.decoder.model.layers[i](x)
        output=x
        return tf.keras.Model(inputs=input,outputs=output,name="JSCCcprsModel1")

    def get_cprs_model2(self):
        input=tf.keras.Input(shape=(_HEIGHT,_WIDTH,_NUM_CHANNELS))
        x=self.encoder.model(input)
        output=self.decoder.model(x)
        return tf.keras.Model(inputs=input,outputs=output,name="JSCCcprsModel2")



    

def save_img(index,image,curSNR,trainSNR):
        image=tf.math.multiply(image,255)
        image=tf.cast(image,dtype=tf.uint8)
        image=tf.image.encode_png(image)
        with tf.io.gfile.GFile('./JSCC/imgs_kodak/{}_img_{}_{}in{}.png'.format(args.channel,index,curSNR,trainSNR), 'wb') as file:
            file.write(image.numpy())


def train():
    dataset=get_dataset(True,args.data_dir_train)
    dataset=dataset.map(lambda x:parse_record(x,tf.float32))
    dataset=dataset.batch(args.batch_size_train)
    #print(dataset)
    train_snrs=[19]
    #train_snrs=[4,9,14,19]
    for train_snr in train_snrs:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        JSCC_model=DeepJSCC(channel_snr=train_snr,conv_depth=args.conv_depth,channel_type=args.channel)
        print("train_snr==",train_snr)
        for epoch in range(args.train_epoch):
            print("in epoch {}".format(epoch))
            for index,dataset_batch in enumerate(dataset):
                dataset_batch=tf.repeat(dataset_batch,args.repeatNum,axis=0)
                with tf.GradientTape() as tape:
                    reconstruct_img=JSCC_model(dataset_batch)
                    loss=mse_loss_fn(dataset_batch,reconstruct_img)
                grads = tape.gradient(loss, JSCC_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, JSCC_model.trainable_weights))
        JSCC_model.testmodel1.save(os.path.join(args.model_dir, 'JSCCModel_{}_in{}_cprs1'.format(args.channel,train_snr)))
        JSCC_model.testmodel1.summary()
        JSCC_model.testmodel2.save(os.path.join(args.model_dir, 'JSCCModel_{}_in{}_cprs2'.format(args.channel,train_snr)))
        JSCC_model.testmodel2.summary()
        JSCC_model.encoder.model.save(os.path.join(args.model_dir, 'JSCCEncoder_{}_in{}_cprs'.format(args.channel,train_snr)))
        JSCC_model.encoder.model.summary()
        JSCC_model.decoder.model.save(os.path.join(args.model_dir, 'JSCCDecoder_{}_in{}_cprs'.format(args.channel,train_snr)))
        JSCC_model.decoder.model.summary()
        
train()




#model = keras.models.load_model('path/to/location')

"""
#test
dataset=get_dataset(True,args.data_dir_train)
dataset=dataset.map(lambda x:parse_record(x,tf.float32))
dataset=dataset.batch(1)
print(dataset)
for index,dataset_batch in enumerate(dataset):
    dataset_batches=tf.repeat(dataset_batch,args.repeatNum,axis=0)
    print(index)
    print(dataset_batches)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
JSCC_model=DeepJSCC(channel_snr=args.train_channel_snr,conv_depth=args.conv_depth,channel_type=args.channel)
for epoch in range(args.train_epoch):
    print("in epoch {}".format(epoch))
    for index,dataset_batch in enumerate(dataset):
        dataset_batch=tf.repeat(dataset_batch,10,axis=0)
        print("index==",index)
        with tf.GradientTape() as tape:
            #print("dataset_batch.shape==",dataset_batch.shape)
            reconstruct_batch=JSCC_model(dataset_batch)
            #print("reconstruct_batch.shape==",reconstruct_batch.shape)
            loss=mse_loss_fn(dataset_batch,reconstruct_batch)

        grads = tape.gradient(loss, JSCC_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, JSCC_model.trainable_weights))

    psnrs=[]
    for snr in range(20):
        JSCC_model.change_channel_snr(snr)
        for index,img in enumerate(dataset):
            reconstruct_img=JSCC_model(img)
            origin_img=tf.squeeze(img)
            reconstruct_img=tf.squeeze(reconstruct_img)
            psnr=tf.image.psnr(origin_img,reconstruct_img,max_val=1.0)
            psnrs.append(psnr)
            save_img(index,reconstruct_img,snr,20)

print(psnrs)
print("ok")
"""



