from collections import deque
import numpy as np


class averageQueue():
    def __init__(self):
        self.queue = deque()
    def getValue(self,data):
        if len(self.queue) > 5:
            if(abs(data - np.mean(self.queue)) <= 2 * np.std(self.queue)):
                self.queue.append(data)
                self.queue.popleft()
                return data
            else:
                self.queue.popleft()
                self.queue.appendleft(data)
                return self.queue[5]
        else:
            self.queue.append(data)
            return data


que = averageQueue()

def tes():
    global que
    for i in range(100):
        que.getValue(i)
