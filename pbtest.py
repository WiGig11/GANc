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
import time
import os

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



model = tf.keras.models.load_model('/chz/res/Semcom/chztestkodak_1/GC_encoder_noQuant_keras_test')
generator = tf.keras.models.load_model('/chz/res/Semcom/chztestkodak_1/GC_generator_keras_test')

def test1(model,generator):
    files = os.listdir('/chz/kodak')
    for file in files:
        path = '/chz/kodak/'+file
        test_image = Image.open(path)
        #test_image = test_image.resize((480,320))
        test_image_array =  np.array(test_image)/255.0
        test_image_array = (2 * test_image_array) - 1
        #print(np.max(test_image_array))
        #print(np.min(test_image_array))
        res= ((test_image_array)+1)*255.0/2
        res= res.astype(np.uint8)
        res= cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/chz/res/test_res/testkodak/ori/single_img{}'.format(file), res)
        input = np.array(test_image)/255.0
        input = tf.convert_to_tensor(input,dtype=tf.float32)
        input = (2 * input) - 1
        input = np.expand_dims(input, axis=0)
        #print(np.max(input))
        #print(np.min(input))
        start=time.time()
        prediction = model.predict(input)  # 使用模型进行预测
        #pdb.set_trace()
        #print(type(prediction))
        decodeTime=time.time()-start
        print("encodeTime=",decodeTime)

        start=time.time()
        res = generator.predict(prediction)
        decodeTime=time.time()-start
        print("decodeTime=",decodeTime)

        decoder_results = (res + 1) / 2
        #print(np.max(decoder_results))
        #print(np.min(decoder_results))
        decoder_results = decoder_results*255.0
        decoder_results = decoder_results.astype(np.uint8)
        decoder_results = np.squeeze(decoder_results, axis=0)
        decoder_results = cv2.cvtColor(decoder_results, cv2.COLOR_RGB2BGR)
        #decoder_results=tf.image.encode_png(decoder_results)
        cv2.imwrite('/chz/res/test_res/testkodak/res/single_test{}'.format(file), decoder_results)

def test2():
    # 假设测试图像的路径
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
    parsed_image_dataset=parsed_image_dataset.shuffle(8).batch(1)
    count = 0
    for image_features in parsed_image_dataset:
        image = image_features['image']  # 这里的'image'键对应于您的图像数据
        #pdb.set_trace()
        #print(image-input)
        #print(np.max(image))
        #print(np.min(image))
        nd = np.array(image)
        nd = np.squeeze(nd,axis=0)
        nd = cv2.cvtColor(nd, cv2.COLOR_RGB2BGR)
        nd = (nd+1)/2
        nd *= 255.
        nd=nd.astype(np.uint8)
        cv2.imwrite('notf2litetest/source{}.jpg'.format(str(count)),nd)
        #print(type(image))
        start = time.time()
        prediction = model.predict(image)  # 使用模型进行预测
        encodetime = time.time() - start
        print("encodeTime=",encodetime)

        start = time.time()
        res = generator.predict(prediction)
        decodetime = time.time() - start
        print("decodeTime=",decodetime)
        decoder_results = np.squeeze(res,axis=0)
        decoder_results = cv2.cvtColor(decoder_results, cv2.COLOR_RGB2BGR)
        decoder_results = (decoder_results+1)/2
        decoder_results *= 255.
        decoder_results=decoder_results.astype(np.uint8)
        cv2.imwrite('notf2litetest/res{}.jpg'.format(str(count)),decoder_results)
        count = count+1


def test3(model,generator):
    img1 = cv2.imread('source7.jpg')
    img2 = cv2.imread('single_img1-017.png')
    img1 = np.expand_dims(img1, axis=0)
    prediction1 = model.predict(img1)  # 使用模型进行预测
    res1 = generator.predict(prediction1)
    decoder_results = (res1 + 1) / 2
    print(np.max(decoder_results))
    print(np.min(decoder_results))
    decoder_results = decoder_results*255.0
    decoder_results = decoder_results.astype(np.uint8)
    decoder_results = np.squeeze(decoder_results, axis=0)
    decoder_results = cv2.cvtColor(decoder_results, cv2.COLOR_RGB2BGR)
    cv2.imwrite('res_test.jpg', decoder_results)
    img2 = np.expand_dims(img2, axis=0)
    prediction1 = model.predict(img2)  # 使用模型进行预测
    res1 = generator.predict(prediction1)
    decoder_results = (res1 + 1) / 2
    print(np.max(decoder_results))
    print(np.min(decoder_results))
    decoder_results = decoder_results*255.0
    decoder_results = decoder_results.astype(np.uint8)
    decoder_results = np.squeeze(decoder_results, axis=0)
    decoder_results = cv2.cvtColor(decoder_results, cv2.COLOR_RGB2BGR)
    cv2.imwrite('res_test2.jpg', decoder_results)


test1(model,generator)
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('source7.jpg')
img2 = cv2.imread('single_img1-017.png')

# 确保图像是相同的数据类型
if img1.dtype != img2.dtype:
    img2 = img2.astype(img1.dtype)

# 计算差异
diff = cv2.absdiff(img1, img2)

# 将图像从BGR转换为RGB
diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示图像
print(diff)
plt.imshow(diff)
plt.title('Difference')
plt.show()
'''

