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

        print("Creating dataloader")

        inputs = []
        for window in np.transpose(windows)[0]:

            window = np.delete(window, 5, 1)    #not sure why this col of 0s is here
            if len(window) < 30:
                window = np.concatenate((window, np.zeros((30-len(window), 5))))
            # #zscore normalization
            #
            window = np.array(window, dtype=np.float32)
            # print(window)
            # for feature in range(5):
            #     #ddof = 1 -> sample std, ddof = 0 -> population std
            #     std = np.std(window[:,feature], ddof=1)
            #     if std==0: std+=0.0000001
            #     avg = np.average(window[:,feature])
            #     #print(avg)
            #     # print(window[:,feature].size)
            #     for value in range(window[:,feature].size):
            #         window[value,feature] = ((window[value,feature]-avg)/std)
            # #     #print('feature ', feature, window[:,feature])


            # #min max normalization
            # for feature in range(5):
            #     min = np.min(window[:,feature])
            #     max = np.max(window[:,feature])
            #     #print(max)
            #     #print(window[:,feature].size)
            #     for value in range(window[:,feature].size):
            #         window[value,feature] = ((window[value,feature]-min)/(max-min))
            #     #print('feature ', feature, window[:,feature])
            # # print(window)
            # # exit(0)
            inputs.append(torch.tensor(window.astype('float32')).to(device))

        self.inputs = inputs
        self.targets = torch.tensor(np.transpose(windows)[1].astype('int32')).to(device)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def D(self):
        return self.inputs[0].size


class LSTMClassifier(torch.nn.Module):

    def __init__(self, D, hidden_size, num_layers, C):
        super(LSTMClassifier, self).__init__()
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
    for epoch in range(args.epochs):
        for i, (mb_x, mb_y) in enumerate(train_loader):

            mb_x = mb_x.to(device)
            mb_y = mb_y.long().to(device)
            
            # Forward pass
            outputs = model(mb_x)
            loss = criterion(outputs, mb_y)            

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                    # running_acc += (dev_y_pred_i == dev_y).sum().data.numpy()
                    running_acc += (dev_y_pred_i == dev_y).sum().item()
                
                    #print(running_acc)
                #TODO*** adjust accuracy for neg ratio
                dev_acc = running_acc / running_devN
                print("%03d.%04d -- accuracy on dev: %.3f" % (epoch,i,dev_acc))


def get_acc(model, test_loader):
    # eval on dedicated test set
    running_acc = 0
    running_testN = 0
    with torch.no_grad():
        for test_update, (test_x, test_y) in enumerate(test_loader):

            test_x = test_x.float()
            test_y = test_y.long()

            testN = len(test_x)
            running_testN += testN
            test_y_pred = model(test_x)
            '''
                Below line gets the index (max returns <value, index> of the maximum value along the 1st dimension,
                which represents how confident the model is that the input was <index> of possible outputs
            '''
            _,test_y_pred_i = torch.max(test_y_pred,1)

            running_acc += (test_y_pred_i == test_y).sum().item()

            #print(running_acc)
    test_acc = running_acc / running_testN
    print("TEST ACCURACY: ", test_acc)




def create_DataSet(neg_windows, pos_windows, args):
    negsize = len(neg_windows)
    data = []
    neg_case_count = len(pos_windows) * args.neg_ratio

    for i in range(neg_case_count):
        index = rand.randint(0, len(neg_windows))
        data.append(neg_windows[index])
        neg_windows=np.delete(neg_windows,index,0) #untested
    data = np.concatenate((data, pos_windows), axis=0) #TODO make sure this stays flat
    np.random.shuffle(data) #shuffle data
    return data 

def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.001]",default=0.001)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 200]",default=100)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int) [default: 128]",default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-kfolds", type=int,\
            help="Number of folds to cross validate between (int) [default: 5]", default=5)
    parser.add_argument("-neg_ratio", type=int,\
            help="Ratio of positive to negative cases (int). Accuracy will rise with higher neg_ratio. [default: 2]", default=2)
    return parser.parse_args()

def main(argv):

    args = parse_all_args()

    pos_windows = np.load('pos_cases.npy', allow_pickle=True)
    neg_windows = np.load('neg_cases.npy', allow_pickle=True)

    #shape <num windows, 3>
    #col 0 is ticker, 1 is np array with <date, open, close, , volume>

    #createSet

    test_pos_size = int(np.floor(len(pos_windows)*.2))
    train_pos_size = len(pos_windows)-test_pos_size

    test_pos_window = []
    for i in range(test_pos_size):
        index = rand.randint(0, len(pos_windows))
        test_pos_window.append(pos_windows[index])
        pos_windows=np.delete(pos_windows,index,0) #untested
    test_window = create_DataSet(neg_windows, test_pos_window, args) #creates test window size 357 (with args.neg_ratio = 20)
    train_dev_window = create_DataSet(neg_windows, pos_windows, args) #creates train window size 1449 (with args.neg_ratio = 20)


    D = 5   # TODO** if we add augmented data increase this

    C = 2   # binary output

    #cross validate
    for i in range(0, args.kfolds-1):

        model = LSTMClassifier(D, 128, 2, C)

        model.to(device)

        dev_start = int(np.floor((i * len(train_dev_window)/args.kfolds)))
        dev_end = int(np.floor(((i + 1) * len(train_dev_window)/args.kfolds)))
        
        #start with dev set being 1st size/args.kfolds chunk
        dev_data = train_dev_window[dev_start : dev_end]
        #train is everything else
        train_data = np.concatenate((train_dev_window[:dev_start], train_dev_window[dev_end:]), axis=0)

        #create loaders
        train_dataset = StockWindows(train_data)
        dev_dataset = StockWindows(dev_data)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.mb, shuffle=True, drop_last=False)
        dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=args.mb, shuffle=False, drop_last=False)

        train(model, train_loader, dev_loader, 30, D, args)

    print("\n\n\n\n\nTEST SET ACCURACY:    \n\n")
    test_dataset = StockWindows(test_window)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.mb, shuffle=False, drop_last=False)
    get_acc(model, test_loader)
    print("\n\nTEST SET ACCURACY ^^ ")

if __name__ == "__main__":
	main(sys.argv)
