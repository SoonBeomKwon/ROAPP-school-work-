import numpy as np
import random
random.seed()

def MinMaxScaler(data):
    numerator=data-np.min(data,0)
    denominator=np.max(data,0)-np.min(data,0)
    #noise term prevents the zero division
    return numerator/(denominator+1e-7)

while(True):
    input()
    sample_t=np.array([random.uniform(3.8-3.4, 3.8+3.4),
                    random.uniform(120.9-32.0, 120.9+32.0),
                    random.uniform(69.1-19.4, 69.1+19.4),
                    random.uniform(20.5-16.0, 20.5+16.0),
                    random.uniform(0, 79.8+115.2),
                    random.uniform(32.0-7.9, 32.0+7.9),
                    random.uniform(0.5-0.3, 0.5+0.3),
                    random.uniform(33.2-11.8, 33.2+11.8)])
    print(sample_t)
    sample_t=MinMaxScaler(sample_t)
    sample_t=np.array([sample_t])
    print(sample_t)
    print(sample_t.shape)
