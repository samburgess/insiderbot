import torch
import torch.nn as nn
import numpy as np
import sys
import numpy.random as rand
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StockWindows(torch.utils.data.Dataset):
    
    def __init__(self, windows):

        super(StockWindows, self).__init__()

        inputs = []
        for window in np.transpose(windows)[0]:
            window = np.delete(window, 5, 1)    #not sure why this col of 0s is here
            if len(window) < 30:
                window = np.concatenate((window, np.zeros((30-len(window), 5))))
            inputs.append(torch.tensor(window.astype('float32')))

        self.inputs = inputs

        self.targets = torch.tensor(np.transpose(windows)[1].astype('int32'))

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def D(self):
        return self.inputs[0].size


class LSTMPredictor(torch.nn.Module):

    def __init__(self, D, hidden_size, num_layers, C):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(D, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size*2, C)
        self.squish = nn.Sigmoid()
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.dense(out[:, -1, :])
        out = self.squish(out)
        return out

def train(model, train_loader, dev_loader, sequence_length, D, args):

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (mb_x, mb_y) in enumerate(train_loader):
            print("mby", mb_y)
            print("mbx", mb_x[0])
            exit()
            # mb_x = mb_x.reshape(-1, sequence_length, D).to(device)
            mb_x = mb_x.to(device)
            mb_y = mb_y.long().to(device)
            
            # print("to device")

            # Forward pass
            outputs = model(mb_x)
            # print("model evald")
            loss = criterion(outputs, mb_y)
            # print("loss calculated")
            

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # print("backprop")
            optimizer.step()
            # print("weights adj")
            # print(i)

            if (i % args.report_freq) == 0:
                # eval on dev once per report_freq
                running_acc = 0
                running_devN = 0
                for dev_update, (dev_x, dev_y) in enumerate(dev_loader):

                    dev_x = dev_x.float()
                    dev_y = dev_y.long()

                    devN = len(dev_x)
                    running_devN += devN
                    dev_y_pred = model(dev_x)
                    #print(dev_y_pred)
                    '''
                        Below line gets the index (max returns <value, index> of the maximum value along the 1st dimension,
                        which represents how confident the model is that the input was <index> of possible outputs
                    '''
                    _,dev_y_pred_i = torch.max(dev_y_pred,1)
                    # print("devypred", dev_y_pred)
                    # print("max(devypred, 1): ", torch.max(dev_y_pred,1))
                    # print("devy: ", dev_y)

                    running_acc += (dev_y_pred_i == dev_y).sum().data.numpy()
                
                    #print(running_acc)
                dev_acc = running_acc / running_devN
                print("%03d.%04d: dev %.3f" % (epoch,i,dev_acc))


# def test_model():
#     # Test the model
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for (mb_x, mb_y) in test_loader:
#             # mb_x = mb_x.reshape(-1, sequence_length, D).to(device)
#             mb_x = mb_x.to(device)
#             mb_y = mb_y.to(device)
#             outputs = model(mb_x)
#             _, predicted = torch.max(outputs.data, 1)
#             total += mb_y.size(0)
#             correct += (predicted == mb_y).sum().item()

#         print('Accuracy: {} %'.format(100 * correct / total)) 

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




def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.0001]",default=0.0001)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 30]",default=30)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 128]",default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    return parser.parse_args()

def main(argv):

    args = parse_all_args()

    spike_windows = np.load('predictData.npy', allow_pickle=True)

    #shape <num windows, 3>
    #col 0 is ticker, 1 is np array with <date, open, close, , volume>

    #createSet
    

    #shuffling might not work because they need to be in order at the moment.. this should change
    test_size = 2500
    dev_size = 2000
    train_size = 8000

    test_window_spikes = []
    dev_window_spikes = []
    train_window_spikes = []


    print("s1")
    #make test set
    for i in range(test_size):
        index = rand.randint(0, len(spike_windows))
        test_window_spikes.append(spike_windows[index])
        spike_windows = np.delete(spike_windows,index,0)
    print("s2")
    #make dev set
    for i in range(dev_size):
        index = rand.randint(0, len(spike_windows))
        dev_window_spikes.append(spike_windows[index])
        spike_windows = np.delete(spike_windows,index,0)
    print("s3")
    #make the train set
    for i in range(train_size):
        index = rand.randint(0, len(spike_windows))
        train_window_spikes.append(spike_windows[index])
        spike_windows = np.delete(spike_windows,index,0)
    print("s4")




    D = 5   # TODO** if we add augmented data increase this

    C = 14   # we want it to predict a window of 14 days idk??

    model = LSTMPredictor(D, 128, 2, C)
    
    #trainsteps
    

    #create loaders
    train_dataset = StockWindows(test_window_spikes)
    dev_dataset = StockWindows(dev_window_spikes)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.mb, shuffle=False, drop_last=False)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=args.mb, shuffle=False, drop_last=False)
    print("s5")
        #layers 
        # L = [[32, 4], [8, 1], [2, 1]]

        # f1 = torch.nn.functional.relu

        # opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, dev_loader, 30, D, args)

    # print("\n\n\n\n\nTEST SET ACCURACY:    \n\n")
    # test_dataset = StockWindows(test_window)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.mb, shuffle=False, drop_last=False)
    # get_acc(model, test_loader)
    # print("\n\nTEST SET ACCURACY ^^ ")

if __name__ == "__main__":
	main(sys.argv)
