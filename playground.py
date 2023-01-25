from multiprocessing import Process, Pipe
import time
import torch
import collections
def f(conn):
    time.sleep(0)
    conn.send(torch.rand([4,5,6]))
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.poll())
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    print(parent_conn.poll())
    p.join()

    print()
    col = collections.deque()
    col.append("3")
    col.pop()
    print(col)