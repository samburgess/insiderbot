import numpy as np
import torch
import matplotlib.pyplot as plt

'''
@ author: Sam Burgess
@ date: May 28 2020
Find distribution of volume spikes in positive cases
'''

pos_windows = np.load('pos_cases.npy', allow_pickle=True)

pos_windows = np.delete(pos_windows, 1, 1) #drop target

spikes = []

for window in pos_windows:
    window = torch.tensor(window[0].astype('float32'))
    # print(window)
    window = torch.transpose(window, 0, 1)[4] #just volumes
    #find biggest daily spike
    #NOTE** we are using 30 day moving avg in neg cases not day to day
    largest_spike = 0
    for i in range(len(window)-1):
        daily_ratio = ( window[i+1].item() / window[i].item() )
        largest_spike = daily_ratio if daily_ratio > largest_spike else largest_spike
    spikes.append(largest_spike)

spikes, _ = torch.tensor(spikes).sort()

# abandoned torch at this point bc it doesnt support negative steps

left = np.array(spikes[::2])
right = np.array(spikes[1::2])
right = np.flip(right)

gaussish = np.concatenate((left, right))

f = plt.figure()
plt.plot(gaussish)
plt.show()

f.savefig("foo.pdf", bbox_inches='tight')

