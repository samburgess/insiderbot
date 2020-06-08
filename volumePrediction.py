import torch
import torch.nn as nn
import numpy as np
import sys
import numpy.random as rand
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



class SpikeWindows(torch.utils.data.Dataset):
    
    def __init__(self, windows):

        super(SpikeWindows, self).__init__()

        self.inputs = []
        self.targets = []

        print("Normalizing data, might take a few minutes ... ")

        for window in windows:
            if len(window) < 44:
                window = np.concatenate((window, np.zeros((44-len(window), 5))))
                

            #zscore normalization
            
            for feature in range(5):
                #ddof = 1 -> sample std, ddof = 0 -> population std
                std = np.std(window[:,feature], ddof=1)
                if std==0: std+=0.0000001
                avg = np.average(window[:,feature])
                #print(avg)
                # print(window[:,feature].size)
                for value in range(window[:,feature].size):
                    window[value,feature] = ((window[value,feature]-avg)/std)
            #     #print('feature ', feature, window[:,feature])


            # #min max normalization
            # for feature in range(5):
            #     min = np.min(window[:,feature])
            #     max = np.max(window[:,feature])
            #     #print(max)
            #     #print(window[:,feature].size)
            #     for value in range(window[:,feature].size):
            #         window[value,feature] = ((window[value,feature]-min)/(max-min))
            #     #print('feature ', feature, window[:,feature])
                
            # #print('feature ', 4, window[:,4])
            self.inputs.append(torch.tensor(window[14:44]))
            self.targets.append(torch.tensor(window[0:14]))

        print("Data normalized")


    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def D(self):
        return self.inputs.size


class LSTMPredictor(torch.nn.Module):

    def __init__(self, D, hidden_size, num_layers, C):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(D, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size*2, C)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.dense(out[:, -1, :])
        return out

def train(model, train_loader, dev_loader, D, sequence_length, args):

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for i, (mb_x, mb_y) in enumerate(train_loader):

            # mb_x = mb_x.reshape(sequence_length, -1, D).to(device)
            mb_x = mb_x.to(device)

            mb_y = mb_y.float().to(device)
            # Forward pass
            for predictionsMade in range(14):
                outputs = model(mb_x) #[minibatch,5] compared to indexed mby[:,13-i,:]
                mb_x = mb_x[:,0:29,:]
                newDayForSlidingWindow = (outputs.unsqueeze(0)).reshape(-1,1,5)
                mb_x = torch.cat((newDayForSlidingWindow,mb_x),1)
                
                #outputs = outputs.reshape(outputs.shape[0], 1, 5) --> model will output a 1,5 tensor

                # print('outputs', outputs)
                # print('mb_y', mb_y)
                # exit(0)
                # print("model evald")
                loss = criterion(outputs, mb_y[:,13-predictionsMade,:]).float() #getting the correct day's data to compare

                # print("loss calculated")
            
                # Backward and optimize
                optimizer.zero_grad()
                if predictionsMade==13:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                # print("backprop")
                optimizer.step()
                # print("weights adj")
                # print(i)

            # if (i % args.report_freq) == 0:
            #     # eval on dev once per report_freq
            #     sum_loss = 0
            #     updates = 0
            #     for dev_update, (dev_x, dev_y) in enumerate(dev_loader):
            #         #print(dev_update)
            #         dev_x = dev_x.float().to(device)
            #         dev_y = dev_y.long().to(device)
            #         dev_y_pred = ((model(dev_x)).unsqueeze(0)).reshape(-1,1,5)
            #         with torch.no_grad():
            #             for predictionsMade in range(14):
                            
                            
            #                 if predictionsMade != 0:
            #                     newDayForSlidingWindow = ((model(dev_x)).unsqueeze(0)).reshape(-1,1,5)
            #                     dev_y_pred = torch.cat((newDayForSlidingWindow, dev_y_pred),1)
            #                     dev_x = dev_x[:,0:29,:]
            #                     dev_x = torch.cat((newDayForSlidingWindow,dev_x),1)
            #                 if predictionsMade == 0:
            #                     dev_x = dev_x[:,0:29,:]
            #                     dev_x = torch.cat((dev_y_pred,dev_x),1)
                            
                        
            #         sum_loss += criterion(dev_y_pred, dev_y).float()
            #         updates += 1
                    
                
            #     dev_loss = sum_loss / updates
            #     print("%03d.%04d: loss on dev: %.3f" % (epoch,i,dev_loss))

            if (i % args.report_freq) == 0:
                # eval on dev once per report_freq
                sum_loss = 0
                updates = 0
                for dev_update, (dev_x, dev_y) in enumerate(dev_loader):
                    #print(dev_update)
                    dev_x = dev_x.float().to(device)
                    dev_y = dev_y.long().to(device)
                    dev_y_pred = (((model(dev_x)[:,4]).unsqueeze(1)).unsqueeze(2)) #should be making a 10, 1, 1 tensor for the start (last 1 is volume data)
                    with torch.no_grad():
                        for predictionsMade in range(14):
                            newDayForSlidingWindow = ((model(dev_x)).unsqueeze(1)) 
                            if predictionsMade != 0:
                                dev_y_pred = torch.cat((newDayForSlidingWindow[:,:,4].unsqueeze(2), dev_y_pred),1)
                            dev_x = dev_x[:,0:29,:]
                            dev_x = torch.cat((newDayForSlidingWindow,dev_x),1)
                            
                    sum_loss += criterion(dev_y_pred, dev_y[:,:,4].unsqueeze(2)).float()
                    updates += 1
                    
                
                dev_loss = sum_loss / updates
                print("%03d.%04d: loss on dev: %.3f" % (epoch,i,dev_loss))


def get_acc(model, test_loader):
    # eval on dedicated test set
    running_acc = 0
    running_testN = 0
    for test_update, (test_x, test_y) in enumerate(test_loader):

        test_x = test_x.float()
        test_y = test_y.long()

        testN = len(test_x)
        running_testN += testN
        test_y_pred = model(test_x)
        #print(test_y_pred)
        '''
            Below line gets the index (max returns <value, index> of the maximum value along the 1st dimension,
            which represents how confident the model is that the input was <index> of possible outputs
        '''
        _,test_y_pred_i = torch.max(test_y_pred,1)
        # print("testypred", test_y_pred)
        # print("max(testypred, 1): ", torch.max(test_y_pred,1))
        # print("testy: ", test_y)

        running_acc += (test_y_pred_i == test_y).sum().data.numpy()

        #print(running_acc)
    test_acc = running_acc / running_testN
    print("TEST ACCURACY: ", test_acc)

'''
    creates dataloader
    @params
        n - iteration (for file name)
        args
        size - number of <30 day, 14 day> windows to be added
        start - index in dataset to start reading from [0..1] - actual index = start*len(data)
        end - index is end*len(data)-1
        shuffle - shuffle dataset or not
'''
def get_data(n, args, start, end, shuffle=True):

    spike_windows = np.load('predictData' + str(n) +'.npy', allow_pickle=True)
    start_i = int(np.floor(start*len(spike_windows)))
    end_i = int(np.floor(end*len(spike_windows)))-1
    #is this grabing less than it should bc-1 ? -alex
    spike_windows = spike_windows[start_i:end_i]
    #shape <num windows, 3>
    #col 0 is ticker, 1 is np array with <date, open, close, , volume>

    # window_spikes = []

    # print("s1")
    # #make test set
    # for i in range(len(spike_windows)):
    #     index = rand.randint(0, len(spike_windows))
    #     window_spikes.append(spike_windows[index])
    #     spike_windows = np.delete(spike_windows,index,0)
    # print("s2")

        
    #create loaders
    dataset = SpikeWindows(spike_windows)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.mb, shuffle=shuffle, drop_last=False)

    return loader



def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.0001]",default=0.1)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 10]",default=10)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 128]",default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-data_splits", type=int,\
            help="number of splits in the data the program should expect", default=5)
    return parser.parse_args()

def main(argv):

    args = parse_all_args()

    D = 5   # TODO** if we add augmented data increase this

    C = 5   # we want it to predict a window of 14 days idk??

    model = LSTMPredictor(D, 32, 5, C)
    model = model.to(device)

    test_loader = get_data(5, args, 0, 1)

    for i in range(1, args.data_splits-1):

        train_loader = get_data(i, args, 0, 0.8)
        dev_loader = get_data(i, args, 0.8, 1)

        train(model, train_loader, dev_loader, D, 60, args)

    # print("\n\n\n\n\nTEST SET ACCURACY:    \n\n") 
    # test_dataset = StockWindows(test_window)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.mb, shuffle=False, drop_last=False)
    # get_acc(model, test_loader)
    # print("\n\nTEST SET ACCURACY ^^ ")

if __name__ == "__main__":
	main(sys.argv)
