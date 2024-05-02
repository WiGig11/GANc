import cv2
import numpy as np
import socket
import struct
import tensorflow as tf
import time
import pickle
decoder_dir='./GANLite_models/GC_generator_wood_high.tflite'

# model init
interpreterD = tf.lite.Interpreter(model_path=decoder_dir)
interpreterD.allocate_tensors()
input_details = interpreterD.get_input_details()
output_details = interpreterD.get_output_details()

# sockey init
HOST='192.168.1.127'
PORT=9999
server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #创建socket对象
server.bind((HOST,PORT))
server.setblocking(0)
print('server initialized')
print('now waiting for frames...')
while True:
    data = None
    try:
        data, _ = server.recvfrom(921600)
        data=pickle.loads(data)
        receive_data = np.frombuffer(data, dtype='uint8').astype(np.float32).reshape(input_details[0]['shape'])
        interpreterD.set_tensor(input_details[0]['index'], receive_data)
        start=time.time()
        interpreterD.invoke()
        end=time.time()
        #print("time==",end-start)
        tflite_results = interpreterD.get_tensor(output_details[0]['index'])
        decoder_results = np.squeeze(tflite_results,axis=0)
        decoder_results = (decoder_results+1)/2
        decoder_results *= 255.
        decoder_results=decoder_results.astype(np.uint8)
        cv2.imshow('server', decoder_results)
    except BlockingIOError as e:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

server.close()
cv2.destroyAllWindows()


