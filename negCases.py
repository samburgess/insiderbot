import pandas as pd
import numpy as np
import os
import sys

SPIKE_SIZE = 12      #how large volume spike from day t-1 to day t needs to be to get added to input. 2 = double volume
WINDOW_SIZE = 30
OFFSET = 3          #days after volume spike to add to window
JUMP_SIZE = 5       #days to jump forward after adding new window



input = []

for filename in os.listdir('Stocks'):

    read = './Stocks/'+filename
    print(filename)

    try:
        data = pd.read_csv(read).to_numpy()

        #iterate thru data, finding windows prior to volume spikes
        for i in range(30, len(data)):
            moving_avg = 0  #volume 
            for j in range(i-30, i):
                moving_avg += data[j][5]
            moving_avg /= 30
            if data[i][5] > moving_avg*SPIKE_SIZE or data[i][5]*SPIKE_SIZE < moving_avg:
                #make window
                window = np.array([data[i-(WINDOW_SIZE-OFFSET) : i+OFFSET], 0])
                window[0] = np.delete(window[0], 0, 1)  #drop date
                window[0] = window[0].astype('float32') #cast stock data to float
                input.append(window)
                i += JUMP_SIZE
    except SystemExit:
        exit(0)
    except:
        print("Couldn't add ", read)
        exit(0)

print("parsed whole set")

np.save('neg_cases.npy', input)