import numpy as np
import pandas as pd
import datetime as dt

WINDOW_SIZE = 30
OFFSET = 3          #days after volume spike to add to window

caseList = pd.read_csv('cases.csv').to_numpy()

#list of windows [ticker, [data]]
windows = []

goof_count = 0

for case in caseList:

    read = './Stocks/'+case[0]+'.us.txt'
    
    try:

        case[1] = dt.datetime.strptime(case[1], '%m/%d/%Y')

        stock = pd.read_csv(read).to_numpy()

        # TODO** replace this loop w pythonic code

        for i in range(len(stock)):
            if dt.datetime.strptime(stock[i][0], '%Y-%m-%d') == case[1]:
                window = np.array([stock[i - (WINDOW_SIZE - OFFSET) : i + OFFSET], 1])  #1 at end for y data
                window[0] = np.delete(window[0], 0, 1)
                windows.append(window)
    except:
        goof_count += 1
        # print("error getting "+case[0]+" on "+str(case[1]))


print(str(goof_count), " goofs")

np.save('pos_cases.npy', windows)