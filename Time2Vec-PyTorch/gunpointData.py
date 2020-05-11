from torch.utils.data import Dataset
import pandas as pd
import numpy as np
#from pyts.datasets import load_gunpoint
import sys
from datasets import load_gunpoint
np.set_printoptions(threshold=sys.maxsize)

class GunPointData(Dataset):
    def __init__(self):
        super(GunPointData, self).__init__()

        #df = pd.read_csv("./data/toy_dataset.csv")
        #X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
        _, train_data, _, train_target = load_gunpoint(return_X_y=True)
        if len(train_target) != len(train_data):
            print("Error dataset x and y not equal")
            exit(1)

        new_train_data = []
        for i in range(len(train_data)):

            new_train_data.append(np.mean(train_data[i]))
        # print("new", new_train_data)
        # print(train_target)
        self.x = new_train_data
        self.y = train_target-1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]


if __name__ == "__main__":
    dataset = GunPointData()
    print(dataset[6])