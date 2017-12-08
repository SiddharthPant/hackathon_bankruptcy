# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:39:25 2017

@author: sidpa
"""
# In[]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[]:
filepath = "bank/bank.csv"
bank_data = pd.read_csv(filepath, sep=';')
bank_data.info()

full_filepath = "bank/bank-full.csv"
bank_full_data = pd.read_csv(full_filepath, sep=';')

# In[]:

# In[]: