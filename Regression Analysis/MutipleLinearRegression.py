#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.getcwd()

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


