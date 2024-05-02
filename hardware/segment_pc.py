import cv2
import numpy as np
import socket
import struct

HOST='192.168.1.127'
PORT=9999
buffSize=65535

server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #创建socket对象
server.bind((HOST,PORT))
server.setblocking(0)
print('server initialized')
print('now waiting for frames...')
while True:
    data = None
    try:
        data, _ = server.recvfrom(921600)
        receive_data = np.frombuffer(data, dtype='uint8')
        r_img = cv2.imdecode(receive_data, 1)
        cv2.imshow('server', r_img)
    except BlockingIOError as e:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
server.close()
cv2.destroyAllWindows()






