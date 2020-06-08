import pandas as pd
import numpy as np
import os
import sys

SPIKE_SIZE = 12      #how large volume spike from day t-1 to day t needs to be to get added to input. 2 = double volume
WINDOW_SIZE = 30
JUMP_SIZE = 5       #days to jump forward after adding new window



input = []
count = 0
n=1
for filename in os.listdir('Stocks'):

    read = './Stocks/'+filename
    #print(filename)

    try:
        data = pd.read_csv(read).to_numpy()
        #iterate thru data, finding windows prior to volume spikes
        for i in range(30, len(data) - 30):
            moving_avg = 0  #volume 
            for j in range(i-30, i):
                moving_avg += data[j][5]
            moving_avg /= 30
            if data[i][5] > moving_avg*SPIKE_SIZE or data[i][5]*SPIKE_SIZE < moving_avg:
                #make window
                count += 1
                
                window = np.array(data[i-14 : i+WINDOW_SIZE])
                window = np.delete(window, 0, 1)  #drop date
                window = np.delete(window, 5, 1)    #not sure why this col of 0s is here
                window = window.astype('float32') #cast stock data to float
                if len(window) != 44:
                    print("Fucked up!", len(window), "\n count: ", count, "\n n: ", n)
                    exit(0)
                input.append(window)
                if(count%50000 == 0):
                    np.save('predictData' + str(n) +'.npy', input)
                    n+=1
                    input=[]
                i += JUMP_SIZE
    except SystemExit:
        exit(0)
    except:
        print("Couldn't add ", read)

print("parsed whole set")

np.save('predictData5.npy', input)
print('predictData.npy created', len(input))