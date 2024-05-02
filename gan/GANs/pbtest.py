import tensorflow as tf

import numpy as np
from PIL import Image
import pdb

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import pdb
import cv2
import os
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
model = tf.keras.models.load_model('//hsx/gan/Semcom/GAN/chztest/GC_encoder_Quant_keras_test')
generator = tf.keras.models.load_model('//hsx/gan/Semcom/GAN/chztest/GC_generator_keras_test')

files = os.listdir('//hsx/Kodak')
for file in files:
    path = '//hsx/Kodak/'+file
    test_image = Image.open(path)
    test_image = test_image.resize((384, 256))
    test_image_array =  np.array(test_image)/255.0
    test_image_array = (2 * test_image_array) - 1
    print(np.max(test_image_array))
    print(np.min(test_image_array))
    res= ((test_image_array)+1)*255.0/2
    res= res.astype(np.uint8)
    res= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite('single_img{}.png'.format(file), res)

    input = np.array(test_image)/255.0
    input = tf.convert_to_tensor(input,dtype=tf.float32)
    input = (2 * input) - 1
    input = np.expand_dims(input, axis=0)
    print(np.max(input))
    print(np.min(input))
    prediction = model.predict(input)  # 使用模型进行预测
    res = generator.predict(prediction)
    decoder_results = (res + 1) / 2
    print(np.max(decoder_results))
    print(np.min(decoder_results))
    decoder_results = decoder_results*255.0
    decoder_results = decoder_results.astype(np.uint8)
    decoder_results = np.squeeze(decoder_results, axis=0)
    decoder_results = cv2.cvtColor(decoder_results, cv2.COLOR_RGB2BGR)
    cv2.imwrite('single_test{}.png'.format(file), decoder_results)


# 假设测试图像的路径
raw_image_dataset = tf.data.TFRecordDataset('kodak.tfrecords')
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
parsed_image_dataset=parsed_image_dataset.shuffle(8).batch(1)
count = 0
for image_features in parsed_image_dataset:
    image = image_features['image']  # 这里的'image'键对应于您的图像数据
    #pdb.set_trace()
    print(image-input)
    print(np.max(image))
    print(np.min(image))
    nd = np.array(image)
    nd = np.squeeze(nd,axis=0)
    nd = cv2.cvtColor(nd, cv2.COLOR_RGB2BGR)
    nd = (nd+1)/2
    nd *= 255.
    nd=nd.astype(np.uint8)
    cv2.imwrite('source{}.jpg'.format(str(count)),nd)
    #print(type(image))
    prediction = model.predict(image)  # 使用模型进行预测
    res = generator.predict(prediction)
    decoder_results = np.squeeze(res,axis=0)
    decoder_results = cv2.cvtColor(decoder_results, cv2.COLOR_RGB2BGR)
    decoder_results = (decoder_results+1)/2
    decoder_results *= 255.
    decoder_results=decoder_results.astype(np.uint8)
    cv2.imwrite('res{}.jpg'.format(str(count)),decoder_results)
    count = count+1


