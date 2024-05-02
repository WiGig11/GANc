# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run image segmentation."""

import pickle
import argparse
import sys
import time
from typing import List
import socket
import cv2
import numpy as np
from tflite_support.task import vision
import tflite_runtime.interpreter as tflite

# Visualization parameters
_FPS_AVERAGE_FRAME_COUNT = 10
_FPS_LEFT_MARGIN = 24  # pixels
_LEGEND_TEXT_COLOR = (0, 0, 255)  # red
_LEGEND_BACKGROUND_COLOR = (255, 255, 255)  # white
_LEGEND_FONT_SIZE = 1
_LEGEND_FONT_THICKNESS = 1
_LEGEND_ROW_SIZE = 20  # pixels
_LEGEND_RECT_SIZE = 16  # pixels
_LABEL_MARGIN = 10
_OVERLAY_ALPHA = 0.5
_PADDING_WIDTH_FOR_LEGEND = 150  # pixels

HOST='192.168.1.127'
PORT=9999


def run(model: str,camera_id: int, width: int, height: int) -> None:

  # Initialize the jscc model.
  interpreterE = tflite.Interpreter(model_path=model)
  interpreterE.allocate_tensors()
  encoder_input_details = interpreterE.get_input_details()
  encoder_output_details = interpreterE.get_output_details()
  # Initialize the socket
  
  try:
    #create an AF_INET, STREAM socket (TCP)
    client=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #socket对象
    client.connect((HOST,PORT))
    print("socket success")
  except socket.error:
    sys.exit('ERROR: socket failed')
  
  
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  # Continuously capture images from the camera and run inference.
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)
    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor_image=np.expand_dims(rgb_image,axis=0)
    interpreterE.set_tensor(encoder_input_details[0]['index'], tensor_image)
    interpreterE.invoke()
    encoder_results = interpreterE.get_tensor(encoder_output_details[0]['index'])
    
    send_data=encoder_results.astype(np.uint8)
    print("send_data==",sys.getsizeof(send_data))
    send_data=pickle.dumps(send_data)
    print("send_data==",sys.getsizeof(send_data))
    # Calculate the FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    # send a tensor
    client.sendto(send_data,(HOST,PORT))
  cap.release()
  client.close()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of image segmentation model.',
      required=False,
      default='GC_encoder_noQuant_origin_C4.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth,args.frameHeight)


if __name__ == '__main__':
  main()
