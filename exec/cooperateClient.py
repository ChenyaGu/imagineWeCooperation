#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:55:18 2020

@author: arthurtan
"""

import socket
import pickle
import sys
import numpy as np
from threading import Thread
from requests import get

try:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 12345
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', port))
    server.listen(5)

    ip = get('https://api.ipify.org').text
    print(f"My public IP address is: {ip}")
    print("Server is listening")
    position = ["", "", "", "", "", ["",""]]
    def threadedClient(client):
        global position
        reply = ''
        while True:
            try:
                data = client.recv(2048)
                if not data:
                    client.send(str.encode("Goodbye"))
                    break
                else:
                    reply = pickle.loads(data)
                    if len(reply) == 2:
                        position[5][1] = reply

                    else:
                        position[0] = reply[0]
                        position[1] = reply[1]
                        position[2] = reply[2]
                        position[3] = reply[3]
                        position[4] = reply[4]
                        position[5][0] = reply[5]
                    client.send(pickle.dumps(position))
            except:
                break

    while True:
        client, address = server.accept()
        print(address)
        print(f"Socket Up and running with a connection from {address}")
        Thread(target = threadedClient, args = (client,)).start()
        print("Awaiting new connection")

except KeyboardInterrupt:
    print("Closing Connection and freeing the port.")
    client.close()
    sys.exit()

