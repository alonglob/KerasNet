import threading
import time

exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, threadID, name, counter, process):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("Starting " + self.name)
        print("Exiting " + self.name)


def print_time():
    print('hello')

# Create new threads
thread1 = myThread(1, "Thread-1", 1, print_time())
thread2 = myThread(2, "Thread-2", 2, print_time())

# Start new Threads
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print("Exiting Main Thread")
