
import socket
import selectors
import types

HOST = '192.168.31.23'
PORT = 12345
sel = selectors.DefaultSelector()
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.bind((HOST, PORT))
lsock.listen()
print("Listening on...", (HOST, PORT))
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)


def accept_wrapper(sock):
    conn, addr = sock.accept()  # should be ready to send
    print("Acepted connection from", addr)
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    print(data)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)


def service_connection(key, mask):
    sock = key.fileobj
    data = key.data
    # print(key,'-----------------')
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)  # should be ready to read
        if recv_data:
            data.outb += recv_data
        else:
            print("Closing connection to...", data.addr)
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        if data.outb:
            print("Echoing", repr(data.outb), "to", data.addr)
            var= bytes("98871", encoding="utf8")
            sent = sock.send(data.outb)  # should be ready to write
            data.outb = data.outb[sent:]


try:
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                accept_wrapper(key.fileobj)
            else:
                service_connection(key, mask)
except KeyboardInterrupt:
    print("Caught keyboard interrrupt, exiting")
finally:
    sel.close()