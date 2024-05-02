import tensorflow as tf
import os

def _bytes_feature(value):
    """返回一个 bytes_list 从一个字符串 / 字节"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image)  # 默认解码为 uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # 转换为浮点类型进行处理
    image = tf.image.resize(image, [256, 384])  # 调整图像大小
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)  # 在编码前转换回 uint8
    image = tf.image.encode_png(image)  # 重新编码为 PNG
    return image

def create_tfrecord(image_paths, tfrecord_file):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for img_path in image_paths:
            image = serialize_image(img_path)
            feature = {
                'image': _bytes_feature(image),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# 假设你的图像都在 'dataset/kodak/' 文件夹中
image_dir = './K19/'
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
tfrecord_file = './k19.tfrecords'

create_tfrecord(image_paths, tfrecord_file)
