import numpy as np
import matplotlib.pyplot as plt
from utils import many_co_cluster_match
import time

qq=np.zeros(shape=50)
for i in range(1):

    A=np.zeros(shape=[200,50])
    for n in range(50):
        A[:,n]=np.random.randint(0,6,200)

    seconds = time.time()
    co_cluster_count = many_co_cluster_match(A)
    print("Seconds since epoch =", time.time()-seconds)
