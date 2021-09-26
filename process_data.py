"""
MIT License

Copyright (c) 2021 Yoga Suhas Kuruba Manjunath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

random.seed(10)

S_T = 50 # number of train samples
N = 20 # size of slice

config =[N,S_T]

lbl = {"label":[],"class":[]}

def get_classes():
        return lbl
 
def chunks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def slice(dfm, chunk_size):
    indices = chunks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

def load_data(data_path,__type):

    ntk_data = []
    
    path, dirs, files = next(os.walk(data_path))
    
    for x, file in enumerate(files):
        D = []
        print(file)

            
        df = pd.read_csv(os.path.join(data_path,file))
        
        if ("Info" in df.columns):
            df = df.drop(columns=["Info"])
        
        if ("No." in df.columns):
            df = df.drop(columns=["No."])
        
        if ("src port" in df.columns):
            df = df.drop(columns=["src port"])
        
        if ("dst port" in df.columns):
            df = df.drop(columns=["dst port"])	

        if ("Protocol" in df.columns):
            df = df.drop(columns=["Protocol"])  

        if ("Source" in df.columns) and ("Destination" in df.columns):
            df = df.drop(columns=["Source", "Destination"])        

        print(len(df))

        df = df.dropna()
                
        #N = 20 # slice size
        
        df_sp = slice(df, config[0])
        print(len(df_sp))
        # for raw data
        
        """
        for i in range(0,len(df_sp)):
            if len(df_sp[i]) == spl:
                t = np.array(df_sp[i])
                t = t.reshape(t.shape[0]*t.shape[1])
                D.append(t)
            
        """
        # for statistical data
        
        if __type == "train":
            s = 0 # start index
            s_t = s+config[1] # for training, first 50 samples are considered.

        elif __type == "test":
            s = config[1] # start index 
            s_t = len(df_sp)# for testing includes all remaining data.


        for i in range(s,s_t):    
            v = []
            up_dn = []
            flow_dur = []
            
            if len(df_sp[i]) == N:
                for j in  range(0, (df_sp[i].shape[1])):                    
                    
                    if j ==0:
                        offset = df_sp[i].iloc[1,0]
                        l = abs(df_sp[i].iloc[:,0] - offset)
                        flow_dur = sum(l)
                    elif j == 1 or j == 2:
                        df_f =(df_sp[i].iloc[:,j].agg(['min','max', 'mean','std']))
                        v.append(np.array(df_f))
                    elif j == 3:
                        up_dn = np.array(df_sp[i].iloc[:,3].value_counts())

                v = np.array(v)
                v = v.reshape(v.shape[0]*v.shape[1])
                k = np.array(up_dn)
                flow_dur = np.array(flow_dur)
                v = np.append(v,up_dn)
                v = np.append(v,flow_dur)
                D.append(v)

        df_t = pd.DataFrame(D)

        df_t["label"] = x
        df_t["class"] = file.split(".")[0]   
        lbl["label"].append(file.split(".")[0])
        lbl["class"].append(x)
        
        ntk_data.append(df_t)         

    ntk_data_label = pd.concat(ntk_data,axis=0, sort=False, ignore_index=True)
    ntk_data_label = ntk_data_label.replace(to_replace = np.nan, value = 0)

    ntk_data_label.columns =['min_length','max_length','mean_length','std_length','min_iat','max_iat','mean_iat','std_iat', 'dn', 'up', 'time','label','class']

    ntk_data_label = ntk_data_label.drop(columns= ["class"])
    X = np.array(ntk_data_label.iloc[:,ntk_data_label.columns !="label"])
    y = np.array(ntk_data_label["label"]) 

    return X,y

# call to get the dataset for train or test    
def get_dataset(__data,__type):
    lbl["label"].clear()
    lbl["class"].clear()

    if __data == "dataset1":
        return load_data('./IEEE_dataport_pre_processed/',__type)
    if __data == "dataset2":
        return load_data('./UNB_pre_processed/',__type)

# To plot histogram    
def hist_plot(data_path):
    path, dirs, files = next(os.walk(data_path))
    
    for x, file in enumerate(files):
        print(file)
            
        df = pd.read_csv(os.path.join(data_path,file))
        
        if ("Info" in df.columns):
            df = df.drop(columns=["Info"])
        
        if ("No." in df.columns):
            df = df.drop(columns=["No."])
        
        if ("src port" in df.columns):
            df = df.drop(columns=["src port"])
        
        if ("dst port" in df.columns):
            df = df.drop(columns=["dst port"])	

        if ("Protocol" in df.columns):
            df = df.drop(columns=["Protocol"])  

        if ("Source" in df.columns) and ("Destination" in df.columns):
            df = df.drop(columns=["Source", "Destination"])        


        df = df.dropna()
                
        N = 20 # N
        
        df_sp = slice(df, N)

        fig = plt.figure(dpi=300)

        ax = fig.subplots(2,3)

        ax[0,0].hist(df_sp[15]['Length'], bins = 100,edgecolor='darkblue', color='darkred', alpha=.5, density=True)
        ax[0,0].legend(["len"])
        #ax.set_title(file.split(".")[0])
        ax[0,0].set_title("a")

        ax[0,1].hist(df_sp[15]['inter-arrival'], bins = 100,edgecolor='darkblue', color='darkred', alpha=.5, density=True)
        ax[0,1].legend(["iat"])
        #ax.set_title(file.split(".")[0])    
        ax[0,1].set_title("b")
        

        ax[0,2].hist(df_sp[15]['Dir'], bins = 100,edgecolor='darkblue', color='darkred', alpha=.5, density=True)
        ax[0,2].legend(["dir"])
        #ax.set_title(file.split(".")[0])                        
        ax[0,2].set_title("c")
        

        ax[1,0].hist(df_sp[115]['Length'], bins = 100,edgecolor='darkblue', color='darkred', alpha=.5, density=True)
        ax[1,0].legend(["len"],)
        #ax.set_title(file.split(".")[0])        
        ax[1,0].set_title("d")


        ax[1,1].hist(df_sp[115]['inter-arrival'], bins = 100,edgecolor='darkblue', color='darkred', alpha=.5, density=True)
        ax[1,1].legend(["iat"])
        #ax.set_title(file.split(".")[0])              
        ax[1,1].set_title("e")
        

        ax[1,2].hist(df_sp[115]['Dir'], bins = 100,edgecolor='darkblue', color='darkred', alpha=.5, density=True)
        ax[1,2].legend(["dir"])
        #ax.set_title(file.split(".")[0])           
        ax[1,2].set_title("f")
        
        fig.tight_layout()
        plt.savefig(file.split(".")[0]+'.png')

            

def plot(__data):
    if __data == "dataset1":
        hist_plot('./IEEE_dataport_pre_processed/') #Dataset I folder
    if __data == "dataset2":
        hist_plot('./UNB_pre_processed/') #Dataset II folder        
     
