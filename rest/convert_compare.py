from sys import getsizeof
import tensorflow as tf
import numpy as np
from tensorflow.keras import models
import tensorflow_model_optimization as tfmot
import tempfile
import zipfile
import os
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_model_optimization.python.core.sparsity.keras.prune_registry import PruneRegistry

raw_image_dataset = tf.data.TFRecordDataset('dataset/city/city_label_inst.tfrecords')
# Create a dictionary describing the features.
image_feature_description = {
    'image_left': tf.io.FixedLenFeature([], tf.string),
    'segmentation_label': tf.io.FixedLenFeature([], tf.string),
    'segmentation_instance': tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):
   # Parse the input tf.Example proto using the dictionary above.
    dataset=tf.io.parse_single_example(example_proto, image_feature_description)
    image_png=tf.io.decode_png(dataset['image_left'])
    img=tf.image.convert_image_dtype(image_png,dtype=tf.float32)
    img=tf.image.resize(img,[512,1024])
    img=2*img-1
    return img

city_dataset = raw_image_dataset.map(_parse_image_function)

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
    return image_png

kodak_dataset = raw_image_dataset.map(_parse_image_function)

raw_image_dataset = tf.data.TFRecordDataset('dataset/wood640480.tfrecords')
# Create a dictionary describing the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
}
def _parse_image_function(example_proto):
   # Parse the input tf.Example proto using the dictionary above.
    dataset=tf.io.parse_single_example(example_proto, image_feature_description)
    image_png=tf.io.decode_png(dataset['image'])
    image_png=tf.image.convert_image_dtype(image_png,dtype=tf.float32)
    return image_png

wood_high_dataset = raw_image_dataset.map(_parse_image_function)

def representative_city_data_gen():
  for input_value in city_dataset.batch(1).take(24):
    yield [input_value]


def representative_kodak_data_gen():
  for input_value in kodak_dataset.batch(1).take(24):
    yield [input_value]

def representative_wood_high_data_gen():
  for input_value in wood_high_dataset.batch(1).take(128):
    yield [input_value]

def get_gzipped_file_size(file):
  # It returns the size of the gzipped model in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)/(1024*1024)

def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)/(1024*1024)

def not_apply_pruning_to_PReLU(layer):
  if not isinstance(layer, tf.keras.layers.PReLU):
    return tfmot.sparsity.keras.prune_low_magnitude(layer,**pruning_params)
  return layer


def not_apply_clustering_to_PReLU(layer):
  if not isinstance(layer, tf.keras.layers.PReLU):
    return cluster_weights(layer, **clustering_params)
  return layer

def not_apply_pruning_to_InstanceNormalization(layer):
  if not isinstance(layer, InstanceNormalization):
    return tfmot.sparsity.keras.prune_low_magnitude(layer,**pruning_params)
  return layer

def not_apply_clustering_to_InstanceNormalization(layer):
  if not isinstance(layer, InstanceNormalization):
    return cluster_weights(layer, **clustering_params)
  return layer


def apply_pruning_to_supported_layers(layer):
        if (
            PruneRegistry.supports(layer)
            or isinstance(layer, tfmot.sparsity.keras.PrunableLayer)
            or hasattr(layer, 'get_prunable_weights')
        ):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer


########################################################################
"""
model_path = "./JSCC_models_cprs/JSCCEncoder_awgn_in19_cprs"
JSCCEncoder = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCEncoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
JSCCEncoder_tflite_quant_model = converter.convert()
JSCCEncoder_tflite_quant_file = './tmp/JSCCEncoder_quant.tflite'
with open(JSCCEncoder_tflite_quant_file, 'wb') as f:
  f.write(JSCCEncoder_tflite_quant_model)
print("JSCCEncoder_quant",get_gzipped_file_size(JSCCEncoder_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(JSCCEncoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
JSCCEncoder_tflite_fullquant_model = converter.convert()
JSCCEncoder_tflite_fullquant_file = './tmp/JSCCEncoder_fullquant.tflite'
with open(JSCCEncoder_tflite_fullquant_file, 'wb') as f:
  f.write(JSCCEncoder_tflite_fullquant_model)
print("JSCCEncoder_fullquant",get_gzipped_file_size(JSCCEncoder_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(JSCCEncoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
JSCCEncoder_tflite_IOfullquant_model = converter.convert()
JSCCEncoder_tflite_IOfullquant_file = './tmp/JSCCEncoder_IOfullquant.tflite'
with open(JSCCEncoder_tflite_IOfullquant_file, 'wb') as f:
  f.write(JSCCEncoder_tflite_IOfullquant_model)
print("JSCCEncoder_IOfullquant",get_gzipped_file_size(JSCCEncoder_tflite_IOfullquant_file))

"""
"""
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    JSCCEncoder,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCEncoder)
model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("JSCCEncoder_pruning",get_gzipped_model_size(model_for_pruning))
print("JSCCEncoder_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/JSCCEncoder_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("JSCCEncoder_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    JSCCEncoder,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCEncoder)
model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("JSCCEncoder_pruning",get_gzipped_model_size(model_for_pruning))
print("JSCCEncoder_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/JSCCEncoder_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("JSCCEncoder_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))
#cluster
def not_apply_clustering_to_PReLU(layer):
  if not isinstance(layer, tf.keras.layers.PReLU):
    return cluster_weights(layer, **clustering_params)
  return layer

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 3,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    JSCCEncoder,
    clone_function=not_apply_clustering_to_PReLU,
)
clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
final_model.summary()
print("JSCCEncoder_clustering",get_gzipped_model_size(clustered_model))
print("JSCCEncoder_clustering_export",get_gzipped_model_size(final_model))
model_for_clustering_tflite_file = './tmp/JSCCEncoder_clustering_8_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clusteringg_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clusteringg_tflite_model)
print("JSCCEncoder_clustering_export",get_gzipped_file_size(model_for_clustering_tflite_file))
"""


########################################################
# convert to tflite 
# work only for export model
"""
model_for_pruning_tflite_file = '/tmp/JSCCEncoder_pruning.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning)
model_for_pruning_tflite_model = converter.convert()
with open(model_for_pruning_tflite_file, 'wb') as f:
  f.write(model_for_pruning_tflite_model)
print("JSCCEncoder_pruning",get_gzipped_file_size(model_for_pruning_tflite_file))

model_for_pruning_export_tflite_file = '/tmp/JSCCEncoder_pruning_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("JSCCEncoder_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

model_for_clustering_tflite_file = '/tmp/JSCCEncoder_clustering.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(clustered_model)
model_for_clustering_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clustering_tflite_model)
print("JSCCEncoder_clustering",get_gzipped_file_size(model_for_pruning_export_tflite_file))

model_for_clustering_tflite_file = '/tmp/JSCCEncoder_clustering_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clusteringg_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clusteringg_tflite_model)
print("JSCCEncoder_clustering_export",get_gzipped_file_size(model_for_clustering_tflite_file))
"""

########################################################################

def representative_decode_dataset_jscc():
    for _ in range(24):
      data = np.random.rand(1, 256, 284, 16)
      yield [data.astype(np.float32)]
"""
model_path = "./JSCC_models_cprs/JSCCDecoder_awgn_in19_cprs"
JSCCDecoder = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
JSCCDecoder_tflite_quant_model = converter.convert()
JSCCDecoder_tflite_quant_file = './tmp/JSCCDecoder_quant.tflite'
with open(JSCCDecoder_tflite_quant_file, 'wb') as f:
  f.write(JSCCDecoder_tflite_quant_model)
print("JSCCDecoder_quant",get_gzipped_file_size(JSCCDecoder_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(JSCCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_decode_dataset_jscc
JSCCDecoder_tflite_fullquant_model = converter.convert()
JSCCDecoder_tflite_fullquant_file = './tmp/JSCCDecoder_fullquant.tflite'
with open(JSCCDecoder_tflite_fullquant_file, 'wb') as f:
  f.write(JSCCDecoder_tflite_fullquant_model)
print("JSCCDecoder_fullquant",get_gzipped_file_size(JSCCDecoder_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(JSCCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_decode_dataset_jscc
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
JSCCDecoder_tflite_IOfullquant_model = converter.convert()
JSCCDecoder_tflite_IOfullquant_file = './tmp/JSCCDecoder_IOfullquant.tflite'
with open(JSCCDecoder_tflite_IOfullquant_file, 'wb') as f:
  f.write(JSCCDecoder_tflite_IOfullquant_model)
print("JSCCDecoder_IOfullquant",get_gzipped_file_size(JSCCDecoder_tflite_IOfullquant_file))
"""
"""
#pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    JSCCDecoder,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCDecoder)
model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("JSCCDecoder_pruning",get_gzipped_model_size(model_for_pruning))
print("JSCCDecoder_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/JSCCDecoder_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("JSCCDecoder_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    JSCCDecoder,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCDecoder)
model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("JSCCDecoder_pruning",get_gzipped_model_size(model_for_pruning))
print("JSCCDecoder_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/JSCCDecoder_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("JSCCDecoder_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#cluster
def not_apply_clustering_to_PReLU(layer):
  if not isinstance(layer, tf.keras.layers.PReLU):
    return cluster_weights(layer, **clustering_params)
  return layer

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 3,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    JSCCDecoder,
    clone_function=not_apply_clustering_to_PReLU,
)
clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
final_model.summary()
print("JSCCDecoder_clustering",get_gzipped_model_size(clustered_model))
print("JSCCDecoder_clustering_export",get_gzipped_model_size(final_model))

model_for_clustering_tflite_file = './tmp/JSCCDecoder_clustering_3_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clusteringg_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clusteringg_tflite_model)
print("JSCCDecoder_clustering_3_export",get_gzipped_file_size(model_for_clustering_tflite_file))
"""
###############################################################
# work
"""
model_path = "./JSCC_models_cprs/JSCCModel_awgn_in19_cprs1"
JSCCModel1 = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
JSCCModel1_tflite_quant_model = converter.convert()
JSCCModel1_tflite_quant_file = './tmp/JSCCModel1_quant.tflite'
with open(JSCCModel1_tflite_quant_file, 'wb') as f:
  f.write(JSCCModel1_tflite_quant_model)
print("JSCCModel1_quant",get_gzipped_file_size(JSCCModel1_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
JSCCModel1_tflite_fullquant_model = converter.convert()
JSCCModel1_tflite_fullquant_file = './tmp/JSCCModel1_fullquant.tflite'
with open(JSCCModel1_tflite_fullquant_file, 'wb') as f:
  f.write(JSCCModel1_tflite_fullquant_model)
print("JSCCModel1_fullquant",get_gzipped_file_size(JSCCModel1_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
JSCCModel1_tflite_IOfullquant_model = converter.convert()
JSCCModel1_tflite_IOfullquant_file = './tmp/JSCCModel1_IOfullquant.tflite'
with open(JSCCModel1_tflite_IOfullquant_file, 'wb') as f:
  f.write(JSCCModel1_tflite_IOfullquant_model)
print("JSCCModel1_IOfullquant",get_gzipped_file_size(JSCCModel1_tflite_IOfullquant_file))

#pruning 50%
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

JSCCModel1_pruning_model = tf.keras.models.clone_model(
    JSCCModel1,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCModel1)
JSCCModel1_pruning_model.summary()
print("JSCCModel1_pruning",get_gzipped_model_size(JSCCModel1_pruning_model))
JSCCModel1_pruning_export_model = tfmot.sparsity.keras.strip_pruning(JSCCModel1_pruning_model)
JSCCModel1_pruning_export_model.summary()
print("JSCCModel1_pruning_export",get_gzipped_model_size(JSCCModel1_pruning_export_model))

JSCCModel1_pruning_export_tflite_file = './tmp/JSCCModel1_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1_pruning_export_model)
JSCCModel1_pruning_export_tflite_model = converter.convert()
with open(JSCCModel1_pruning_export_tflite_file, 'wb') as f:
  f.write(JSCCModel1_pruning_export_tflite_model)
print('Saved clustered TFLite model to:', JSCCModel1_pruning_export_tflite_file)
print("JSCCModel1_pruning_export_tflite",get_gzipped_file_size(JSCCModel1_pruning_export_tflite_file))



pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

JSCCModel1_pruning_model = tf.keras.models.clone_model(
    JSCCModel1,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCModel1)
JSCCModel1_pruning_model.summary()
print("JSCCModel1_pruning",get_gzipped_model_size(JSCCModel1_pruning_model))
JSCCModel1_pruning_export_model = tfmot.sparsity.keras.strip_pruning(JSCCModel1_pruning_model)
JSCCModel1_pruning_export_model.summary()
print("JSCCModel1_pruning_export",get_gzipped_model_size(JSCCModel1_pruning_export_model))

JSCCModel1_pruning_export_tflite_file = './tmp/JSCCModel1_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1_pruning_export_model)
JSCCModel1_pruning_export_tflite_model = converter.convert()
with open(JSCCModel1_pruning_export_tflite_file, 'wb') as f:
  f.write(JSCCModel1_pruning_export_tflite_model)
print('Saved clustered TFLite model to:', JSCCModel1_pruning_export_tflite_file)
print("JSCCModel1_pruning_export_tflite",get_gzipped_file_size(JSCCModel1_pruning_export_tflite_file))


#cluster
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
JSCCModel1_clustering_model = tf.keras.models.clone_model(
    JSCCModel1,
    clone_function=not_apply_clustering_to_PReLU,
)
JSCCModel1_clustering_model.summary()
print("JSCCModel1_clustering",get_gzipped_model_size(JSCCModel1_clustering_model))
JSCCModel1_clustering_export_model = tfmot.clustering.keras.strip_clustering(JSCCModel1_clustering_model)
JSCCModel1_clustering_export_model.summary()
print("JSCCModel1_clustering_export",get_gzipped_model_size(JSCCModel1_clustering_export_model))
JSCCModel1_clustering_export_tflite_file = './tmp/JSCCModel1_clustering_8_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1_clustering_export_model)
JSCCModel1_clustering_export_tflite_model = converter.convert()
with open(JSCCModel1_clustering_export_tflite_file, 'wb') as f:
  f.write(JSCCModel1_clustering_export_tflite_model)
print('Saved clustered TFLite model to:', JSCCModel1_clustering_export_tflite_file)
print("JSCCModel1_clustering_export_tflite",get_gzipped_file_size(JSCCModel1_clustering_export_tflite_file))
"""
#############################################################
# do not work when init twice?
"""
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
JSCCModel1_clustering_model = tf.keras.models.clone_model(
    JSCCModel1,
    clone_function=not_apply_clustering_to_PReLU,
)
JSCCModel1_clustering_model.summary()
print("JSCCModel1_clustering",get_gzipped_model_size(JSCCModel1_clustering_model))
JSCCModel1_clustering_export_model = tfmot.clustering.keras.strip_clustering(JSCCModel1_clustering_model)
JSCCModel1_clustering_export_model.summary()
print("JSCCModel1_clustering_export",get_gzipped_model_size(JSCCModel1_clustering_export_model))
JSCCModel1_clustering_export_tflite_file = './tmp/JSCCModel1_clustering_8_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel1_clustering_export_model)
JSCCModel1_clustering_export_tflite_model = converter.convert()
with open(JSCCModel1_clustering_export_tflite_file, 'wb') as f:
  f.write(JSCCModel1_clustering_export_tflite_model)
print('Saved clustered TFLite model to:', JSCCModel1_clustering_export_tflite_file)
print("JSCCModel1_clustering_export_tflite",get_gzipped_file_size(JSCCModel1_clustering_export_tflite_file))
"""
#############################################################
# do not work
"""
model_path = "./JSCC/saveModel/JSCCModel_awgn_in19_cprs2"
JSCCModel2 = tf.keras.models.load_model(model_path)
#quant
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
JSCCModel2_tflite_quant_model = converter.convert()
JSCCModel2_tflite_quant_file = './tmp/JSCCModel2_quant.tflite'
with open(JSCCModel2_tflite_quant_file, 'wb') as f:
  f.write(JSCCModel2_tflite_quant_model)
print("JSCCModel2_quant",get_gzipped_file_size(JSCCModel2_tflite_quant_file))
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
JSCCModel2_tflite_fullquant_model = converter.convert()
JSCCModel2_tflite_fullquant_file = './tmp/JSCCModel2_fullquant.tflite'
with open(JSCCModel2_tflite_fullquant_file, 'wb') as f:
  f.write(JSCCModel2_tflite_fullquant_model)
print("JSCCModel2_fullquant",get_gzipped_file_size(JSCCModel2_tflite_fullquant_file))
converter = tf.lite.TFLiteConverter.from_keras_model(JSCCModel2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
JSCCModel2_tflite_IOfullquant_model = converter.convert()
JSCCModel2_tflite_IOfullquant_file = './tmp/JSCCModel2_IOfullquant.tflite'
with open(JSCCModel2_tflite_IOfullquant_file, 'wb') as f:
  f.write(JSCCModel2_tflite_IOfullquant_model)
print("JSCCModel2_IOfullquant",get_gzipped_file_size(JSCCModel2_tflite_IOfullquant_file))

#pruning
model_for_pruning = tf.keras.models.clone_model(
    JSCCModel2,
    clone_function=not_apply_pruning_to_PReLU,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCModel2)
model_for_pruning.summary()
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.summary()
print("JSCCModel2_pruning",get_gzipped_model_size(model_for_pruning))
print("JSCCModel2_pruning_export",get_gzipped_model_size(model_for_pruning))

#cluster
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 3,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    JSCCModel2,
    clone_function=not_apply_clustering_to_PReLU,
)
clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
final_model.summary()
print("JSCCModel2_clustering",get_gzipped_model_size(clustered_model))
print("JSCCModel2_clustering_export",get_gzipped_model_size(final_model))
"""

#####################################################################
# work
"""
model_path = "GC_encoder_noQuant_keras_test"
GCEncodernoQ = tf.keras.models.load_model(model_path)
#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCEncodernoQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCEncodernoQ_tflite_quant_model = converter.convert()
GCEncodernoQ_tflite_quant_file = 'GCEncodernoQ_quant.tflite'
with open(GCEncodernoQ_tflite_quant_file, 'wb') as f:
  f.write(GCEncodernoQ_tflite_quant_model)
print("GCEncodernoQ_quant",get_gzipped_file_size(GCEncodernoQ_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCEncodernoQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
GCEncodernoQ_tflite_fullquant_model = converter.convert()
GCEncodernoQ_tflite_fullquant_file = 'GCEncodernoQ_fullquant.tflite'
with open(GCEncodernoQ_tflite_fullquant_file, 'wb') as f:
  f.write(GCEncodernoQ_tflite_fullquant_model)
print("GCEncodernoQ_fullquant",get_gzipped_file_size(GCEncodernoQ_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCEncodernoQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
GCEncodernoQ_tflite_IOfullquant_model = converter.convert()
GCEncodernoQ_tflite_IOfullquant_file = 'GCEncodernoQ_IOfullquant.tflite'
with open(GCEncodernoQ_tflite_IOfullquant_file, 'wb') as f:
  f.write(GCEncodernoQ_tflite_IOfullquant_model)
print("GCEncodernoQ_IOfullquant",get_gzipped_file_size(GCEncodernoQ_tflite_IOfullquant_file))



#pruning
model_for_pruning = tf.keras.models.clone_model(
    GCEncodernoQ,
    clone_function=not_apply_pruning_to_InstanceNormalization,
)

#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(GCEncodernoQ)
model_for_pruning.summary()
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.summary()
print("GCEncodernoQ_pruning",get_gzipped_model_size(model_for_pruning))
print("GCEncodernoQ_pruning_export",get_gzipped_model_size(model_for_pruning))

#cluster
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 3,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    GCEncodernoQ,
    clone_function=not_apply_clustering_to_InstanceNormalization,
)
clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
final_model.summary()
print("GCEncodernoQ_clustering",get_gzipped_model_size(clustered_model))
print("GCEncodernoQ_clustering_export",get_gzipped_model_size(final_model))
"""

#############################################################
#work

model_path = "GC_encoder_Quant_keras_chz"
GCEncoderQ = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCEncoderQ_tflite_quant_model = converter.convert()
GCEncoderQ_tflite_quant_file = 'GCEncoderQ_quant_chz.tflite'
with open(GCEncoderQ_tflite_quant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_quant_model)
print("GCEncoderQ_quant",get_gzipped_file_size(GCEncoderQ_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
GCEncoderQ_tflite_fullquant_model = converter.convert()
GCEncoderQ_tflite_fullquant_file = 'GCEncoderQ_fullquant_chz.tflite'
with open(GCEncoderQ_tflite_fullquant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_fullquant_model)
print("GCEncoderQ_fullquant",get_gzipped_file_size(GCEncoderQ_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
GCEncoderQ_tflite_IOfullquant_model = converter.convert()
GCEncoderQ_tflite_IOfullquant_file = 'GCEncoderQ_IOfullquant_chz.tflite'
with open(GCEncoderQ_tflite_IOfullquant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_IOfullquant_model)
print("GCEncoderQ_IOfullquant",get_gzipped_file_size(GCEncoderQ_tflite_IOfullquant_file))
"""
#pruning 75
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCEncoderQ,
    clone_function=not_apply_pruning_to_InstanceNormalization,
)

model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("GCEncoderQ_pruning",get_gzipped_model_size(model_for_pruning))
print("GCEncoderQ_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/GCEncoderQ_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCEncoderQ_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#pruning 50
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCEncoderQ,
    clone_function=not_apply_pruning_to_InstanceNormalization,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCEncoder)
model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("GCEncoderQ_pruning",get_gzipped_model_size(model_for_pruning))
print("GCEncoderQ_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/GCEncoderQ_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCEncoderQ_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#cluster

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 3,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    GCEncoderQ,
    clone_function=not_apply_clustering_to_InstanceNormalization,
)
clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
final_model.summary()
print("GCEncoderQ_clustering",get_gzipped_model_size(clustered_model))
print("GCEncoderQ_clustering_export",get_gzipped_model_size(final_model))
model_for_clustering_tflite_file = './tmp/GCEncoderQ_clustering_3_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clustering_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clustering_tflite_model)
print("GCEncoderQ_clustering_export",get_gzipped_file_size(model_for_clustering_tflite_file))
"""
#######################################################################
#work

def representative_decode_dataset_gen():
    for _ in range(24):
      data = np.random.randint(low=-2,high=3,size=(1, 16, 24, 8))
      yield [data.astype(np.float32)]

model_path = "GC_generator_keras_chz"
GCDecoder = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCDecoder_tflite_quant_model = converter.convert()
GCDecoder_tflite_quant_file = 'GCDecoder_quant_chz.tflite'
with open(GCDecoder_tflite_quant_file, 'wb') as f:
  f.write(GCDecoder_tflite_quant_model)
print("GCDecoder_quant",get_gzipped_file_size(GCDecoder_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_decode_dataset_gen
GCDecoder_tflite_fullquant_model = converter.convert()
GCDecoder_tflite_fullquant_file = 'GCDecoder_fullquant_chz.tflite'
with open(GCDecoder_tflite_fullquant_file, 'wb') as f:
  f.write(GCDecoder_tflite_fullquant_model)
print("GCDecoder_fullquant",get_gzipped_file_size(GCDecoder_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_decode_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
GCDecoder_tflite_IOfullquant_model = converter.convert()
GCDecoder_tflite_IOfullquant_file = 'GCDecoder_IOfullquant_chzs.tflite'
with open(GCDecoder_tflite_IOfullquant_file, 'wb') as f:
  f.write(GCDecoder_tflite_IOfullquant_model)
print("GCDecoder_IOfullquant",get_gzipped_file_size(GCDecoder_tflite_IOfullquant_file))
"""
#pruning 75
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCDecoder,
    clone_function=apply_pruning_to_supported_layers,
)

#model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
#model_for_pruning_export.summary()
print("GCDecoder_pruning",get_gzipped_model_size(model_for_pruning))
print("GCDecoder_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/GCDecoder_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCDecoder_pruning75_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#pruning 50
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCDecoder,
    clone_function=apply_pruning_to_supported_layers,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCEncoder)
##model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
#model_for_pruning_export.summary()
print("GCDecoder_pruning",get_gzipped_model_size(model_for_pruning))
print("GCDecoder_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/GCDecoder_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCDecoder_pruning50_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#cluster

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 3,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    GCDecoder,
    clone_function=not_apply_clustering_to_InstanceNormalization,
)
#clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
#final_model.summary()
print("GCDecoder_clustering",get_gzipped_model_size(clustered_model))
print("GCDecoder_clustering_export",get_gzipped_model_size(final_model))
model_for_clustering_tflite_file = './tmp/GCDecoder_clustering_3_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clustering_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clustering_tflite_model)
print("GCDecoder_clustering3_export",get_gzipped_file_size(model_for_clustering_tflite_file))
"""

##############################################################
# work
"""
model_path = "./GAN_models_cprs/GCModel_keras_test"
GCModel = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCModel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCModel_tflite_quant_model = converter.convert()
GCModel_tflite_quant_file = './tmp/GCModel_quant.tflite'
with open(GCModel_tflite_quant_file, 'wb') as f:
  f.write(GCModel_tflite_quant_model)
print("GCModel_quant",get_gzipped_file_size(GCModel_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCModel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
GCModel_tflite_fullquant_model = converter.convert()
GCModel_tflite_fullquant_file = './tmp/GCModel_fullquant.tflite'
with open(GCModel_tflite_fullquant_file, 'wb') as f:
  f.write(GCModel_tflite_fullquant_model)
print("GCModel_fullquant",get_gzipped_file_size(GCModel_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCModel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_kodak_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
GCModel_tflite_IOfullquant_model = converter.convert()
GCModel_tflite_IOfullquant_file = './tmp/GCModel_IOfullquant.tflite'
with open(GCModel_tflite_IOfullquant_file, 'wb') as f:
  f.write(GCModel_tflite_IOfullquant_model)
print("GCModel_IOfullquant",get_gzipped_file_size(GCModel_tflite_IOfullquant_file))
"""
"""
#pruning 75
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCModel,
    clone_function=apply_pruning_to_supported_layers,
)

#model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
#model_for_pruning_export.summary()
print("GCModel_pruning",get_gzipped_model_size(model_for_pruning))
print("GCModel_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/GCModel_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCModel_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#pruning 50
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCModel,
    clone_function=apply_pruning_to_supported_layers,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCEncoder)
#model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
#model_for_pruning_export.summary()
print("GCModel_pruning",get_gzipped_model_size(model_for_pruning))
print("GCModel_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './tmp/GCModel_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCModel_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#cluster
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    GCModel,
    clone_function=not_apply_clustering_to_InstanceNormalization,
)
#clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
#final_model.summary()
print("GCModel_clustering",get_gzipped_model_size(clustered_model))
print("GCModel_clustering_export",get_gzipped_model_size(final_model))
model_for_clustering_tflite_file = './tmp/GCModel_clustering_8_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clustering_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clustering_tflite_model)
print("GCModel_clustering8_export",get_gzipped_file_size(model_for_clustering_tflite_file))

"""

"""
model_path = "./GANLite_models/GC_encoder_Quant_wood_high"
GCEncoderQ = tf.keras.models.load_model(model_path)

#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCEncoderQ_tflite_quant_model = converter.convert()
GCEncoderQ_tflite_quant_file = './GANLite_models/GCEncoderQ_quant_wood_high.tflite'
with open(GCEncoderQ_tflite_quant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_quant_model)
print("GCEncoderQ_quant",get_gzipped_file_size(GCEncoderQ_tflite_quant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_wood_high_data_gen
GCEncoderQ_tflite_fullquant_model = converter.convert()
GCEncoderQ_tflite_fullquant_file = './GANLite_models/GCEncoderQ_fullquant_wood_high.tflite'
with open(GCEncoderQ_tflite_fullquant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_fullquant_model)
print("GCEncoderQ_fullquant",get_gzipped_file_size(GCEncoderQ_tflite_fullquant_file))

converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_wood_high_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8
GCEncoderQ_tflite_IOfullquant_model = converter.convert()
GCEncoderQ_tflite_IOfullquant_file = './GANLite_models/GCEncoderQ_IOfullquant_wood_high.tflite'
with open(GCEncoderQ_tflite_IOfullquant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_IOfullquant_model)
print("GCEncoderQ_IOfullquant",get_gzipped_file_size(GCEncoderQ_tflite_IOfullquant_file))

#pruning 75
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.75, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCEncoderQ,
    clone_function=not_apply_pruning_to_InstanceNormalization,
)

model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("GCEncoderQ_pruning",get_gzipped_model_size(model_for_pruning))
print("GCEncoderQ_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './GANLite_models/GCEncoderQ_wood_high_pruning_75%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCEncoderQ_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#pruning 50
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}
model_for_pruning = tf.keras.models.clone_model(
    GCEncoderQ,
    clone_function=not_apply_pruning_to_InstanceNormalization,
)
#model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(JSCCEncoder)
model_for_pruning.summary()
model_for_pruning_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_pruning_export.summary()
print("GCEncoderQ_pruning",get_gzipped_model_size(model_for_pruning))
print("GCEncoderQ_pruning_export",get_gzipped_model_size(model_for_pruning))
model_for_pruning_export_tflite_file = './GANLite_models/GCEncoderQ_wood_high_pruning_50%_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning_export)
model_for_pruning_export_tflite_model = converter.convert()
with open(model_for_pruning_export_tflite_file, 'wb') as f:
  f.write(model_for_pruning_export_tflite_model)
print("GCEncoderQ_pruning_export",get_gzipped_file_size(model_for_pruning_export_tflite_file))

#cluster

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}
clustered_model = tf.keras.models.clone_model(
    GCEncoderQ,
    clone_function=not_apply_clustering_to_InstanceNormalization,
)
clustered_model.summary()
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
final_model.summary()
print("GCEncoderQ_clustering",get_gzipped_model_size(clustered_model))
print("GCEncoderQ_clustering_export",get_gzipped_model_size(final_model))
model_for_clustering_tflite_file = './GANLite_models/GCEncoderQ_wood_high_clustering_8_export.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
model_for_clustering_tflite_model = converter.convert()
with open(model_for_clustering_tflite_file, 'wb') as f:
  f.write(model_for_clustering_tflite_model)
print("GCEncoderQ_clustering_export",get_gzipped_file_size(model_for_clustering_tflite_file))
"""



#model_path = "./GANLite_models/GC_encoder_Quant_wood_high"
#GCEncoderQ = tf.keras.models.load_model(model_path)
"""
#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCEncoderQ_tflite_quant_model = converter.convert()
GCEncoderQ_tflite_quant_file = './GANLite_models/GCEncoderQ_quant_wood_high.tflite'
with open(GCEncoderQ_tflite_quant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_quant_model)
print("GCEncoderQ_quant",get_gzipped_file_size(GCEncoderQ_tflite_quant_file))

model_path = "./GANLite_models/GC_generator_wood_high"
GCDecoder = tf.keras.models.load_model(model_path)
#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCDecoder_tflite_quant_model = converter.convert()
GCDecoder_tflite_quant_file = './GANLite_models/GC_generator_wood_high.tflite'
with open(GCDecoder_tflite_quant_file, 'wb') as f:
  f.write(GCDecoder_tflite_quant_model)
print("GCDecoder_quant",get_gzipped_file_size(GCDecoder_tflite_quant_file))
"""

#model_path = "./GANLite_models/GC_encoder_Quant_wood_low"
#GCEncoderQ = tf.keras.models.load_model(model_path)
'''
#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCEncoderQ)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCEncoderQ_tflite_quant_model = converter.convert()
GCEncoderQ_tflite_quant_file = './GANLite_models/GCEncoderQ_quant_wood_low.tflite'
with open(GCEncoderQ_tflite_quant_file, 'wb') as f:
  f.write(GCEncoderQ_tflite_quant_model)
print("GCEncoderQ_quant",get_gzipped_file_size(GCEncoderQ_tflite_quant_file))

model_path = "./GANLite_models/GC_generator_wood_low"
GCDecoder = tf.keras.models.load_model(model_path)
#quant
converter = tf.lite.TFLiteConverter.from_keras_model(GCDecoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
GCDecoder_tflite_quant_model = converter.convert()
GCDecoder_tflite_quant_file = './GANLite_models/GC_generator_quant_wood_low.tflite'
with open(GCDecoder_tflite_quant_file, 'wb') as f:
  f.write(GCDecoder_tflite_quant_model)
print("GCDecoder_quant",get_gzipped_file_size(GCDecoder_tflite_quant_file))
'''
