import pandas as pd 
import numpy as np 
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # possible choice

class TrainTestLoader:
    def __init__(self, repeat, developing_size, 
                 evaluation_size, test_size, 
                 scaler_choice, feas_dict,
                 normaliza_target 
                ):
        self.repeat = repeat
        self.developing_size = developing_size
        self.evaluation_size = evaluation_size
        self.test_size = test_size
        self.scaler_choice = scaler_choice
        self.feas_dict = feas_dict
        self.normaliza_target = normaliza_target
        print("features_dict", feas_dict)

    def data_standard(self, df):
        fea_ls = list(df.columns)
        fea_ls.remove('node_ix') # without node ix
        fea_ls.remove('type')
        fea_ls.remove('xy')
        fea_ls.remove('target')
        fea_ls.remove('fea_xy')
        fea_ls.remove("uni_id")
        print("scale fea_ls:", fea_ls)
        if self.scaler_choice is not None:
            if self.scaler_choice == 'minmax':
                scaler = MinMaxScaler() # for gwr make sure there are not all 0 
                print(df[fea_ls].head())
                scaler.fit(np.array(df[fea_ls]))
                df[fea_ls] = scaler.transform(np.array(df[fea_ls]))
            if self.scaler_choice == 'standard':
                scaler = StandardScaler()
                scaler.fit(np.array(df[fea_ls]))
                df[fea_ls] = scaler.transform(np.array(df[fea_ls]))
            if self.scaler_choice == 'combine':
                assert self.feas_dict is not None # make sure that it has features
                stand_feas = self.feas_dict["graph_feas"]
                scaler = StandardScaler()
                scaler_minmax = MinMaxScaler()
                # standard
                scaler.fit(np.array(df[stand_feas]))
                df[stand_feas] = scaler.transform(np.array(df[stand_feas])) 
                # minmax
                f_ls = np.setdiff1d(fea_ls, stand_feas)
                scaler_minmax.fit(np.array(df[f_ls]))
                df[f_ls] = scaler_minmax.transform(np.array(df[f_ls]))  
        if self.normaliza_target:
            scaler = StandardScaler()
            scaler.fit(np.array(df[["target"]]))
            df['target'] = scaler.transform(np.array(df[["target"]]))
        return df 
    
    def simple_train_test(self, df): # only split once
        df = self.data_standard(df)
        node_ix = np.array(df["node_ix"])
        random.shuffle(node_ix)
        developing_array = node_ix[:int(len(node_ix)*self.developing_size)]
        evaluation_array = node_ix[int(len(node_ix)*self.developing_size): int(len(node_ix)*(self.evaluation_size + self.developing_size))]
        test_array = node_ix[int(len(node_ix)*(self.evaluation_size + self.developing_size)): ]
        # split the dataset into three set 
        d = []
        for i in df["node_ix"]:
            if i in developing_array:
                d.append(0)
            elif i in evaluation_array:
                d.append(1)
            elif i in test_array:
                d.append(2)
            else:
                raise IndexError
        df["data_split"] = d
        return df 
