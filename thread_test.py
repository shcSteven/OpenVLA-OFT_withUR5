
import threading
import time# Define two functions that will run in separate threads

def function1():
    while True:
        #print(function2)
        print("Function 1")
        time.sleep(0.5)  # Simulate some workdef function2():
def function2():
    while True:
        #print(function1())
        print("Function 2")
        time.sleep(1)  # Simulate some work# Create threads for each function
thread1 = threading.Thread(target=function1)
thread2 = threading.Thread(target=function2)# Start the threads
thread1.start()
thread2.start()# Keep the main thread running indefinitely
#while True:
    #time.sleep(1)
thread1.join()
thread2.join()# The main thread will never reach this point