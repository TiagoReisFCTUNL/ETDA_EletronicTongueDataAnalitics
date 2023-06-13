# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:46:25 2022

@author: Asus
"""

#IMPORT
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from math import pi

#DEF
def line_func(x, a, b):
    return a*x+b;

def invert(array):
    return array[::-1]

#VALS
first_concentration = 9 #min{|log(concentration)|}
    #select aditional plts
plt_each_signal = False #if True creates a Impedance(frequency) plt for each concentrations
plt_all = False #if True creates a Impedance(frequency) plt with all the concentrations and loops 
plt_PCAs = True #if True creates 2 plts with PCA of the files in folders
plt_HCAs = False #if True creates 3 plts with HCA of the files in folders
    #select what values to analyze
Impedance = True
phase = True
rImpedance = False #R{Z}
iImpedance = False #I{Z}
    #select how to plt values
plt_log_scale = False




#read files
filenames = os.listdir()
folder = os.path.basename(os.getcwd())
temp = []
for f in filenames:
    if f[f.find('.'):] == ".csv":
        temp.append(f)
filenames = temp.copy()
csv_to_open = []
signals = []
temp = filenames.copy()
for s in filenames:
    if s[-14:]=='_corrected.csv':
        temp.remove(s)
    else:
        curr_signal = ""
        if s.find('_') != -1:
            curr_signal = s[:s.find('_')]
        else:
            curr_signal = s[:-4]
        signals.append(curr_signal)
        csv_to_open.append(curr_signal+'_corrected.csv')

#ready csv files
filenames = temp
print('csv_to_open:')
print(csv_to_open)
print('signals:')
print(signals)
print('filenames:')
print(filenames)
columns = "Result Number;Sweep Number;Point Number;Time;frequencyuency (Hz);AC Level (V);DC Level (V);Set Point ('C);Temperature ('C);Control ('C);Impedance Magnitude (Ohms);Impedance Phase Degrees (');Admittance Magnitude (S);Capacitance Magnitude (F);"
for i, file in enumerate(filenames):
    with open(csv_to_open[i], 'w') as f_corrected:
        with open(file) as f:
            contents = f.readlines()
        doc_len = len(contents)
        print(file)
        contents[2] = columns
        for i in range(3):
            f_corrected.writelines(contents[i])
        for i in range(doc_len - 3):
            f_corrected.writelines(contents[i + 3].replace(',','.'))

num_signals = len(csv_to_open)
signals_out = []
signals_out_phase = []
for i in range(num_signals):
    csv = (pd.read_csv(csv_to_open[i], skiprows=2, sep=";")).to_numpy()
    signals_out.append(csv[:,10])
    signals_out_phase.append(csv[:,11])
signal_lenght = csv[:,1].tolist().count(1)
frequency = (pd.read_csv(csv_to_open[0], skiprows=2, sep=";")).to_numpy()[:signal_lenght,4]

total_lenght = signals_out[0].size
plts_file = int((signals_out[0].size)/signal_lenght)
concentration = np.arange(start=first_concentration, stop=first_concentration+num_signals, step=1)

print("signals_out size = " + str(signals_out[0].size))
print("signal_lenght" + str(signal_lenght))
print("plts_file" + str(plts_file))
print("num_signals" + str(num_signals))
print("concentration:")
print(concentration)

#invert
frequency = invert(frequency)
print('frequency:')
print(frequency)
for i in range(num_signals):
    signals_out[i] = invert(signals_out[i])
    signals_out_phase[i] = invert(signals_out_phase[i])

font = {'size'   : 8}

#convert degrees to rads
for i in range(num_signals):
    for f in range(signal_lenght):
        signals_out_phase[i][f] = signals_out_phase[i][f]*(pi/180)

#output each
if(plt_each_signal):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_signals):
        for plt_i in range(plts_file):
            plt.plot(frequency, signals_out[i][plt_i*signal_lenght:(plt_i+1)*signal_lenght], '.', label=str(signals[i]))
        plt.legend()
        plt.title('impedance')
        plt.xlabel('frequency')
        plt.xscale("log")
        plt.ylabel('|impedance| [hom]')
        if(plt_log_scale):
            plt.yscale("log")
        plt.rc('font', **font)
        plt.show()

#output everything
if(plt_all):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for plt_i in range(plts_file):
        for i in range(num_signals):
            plt.plot(frequency, signals_out[i][plt_i*signal_lenght:(plt_i+1)*signal_lenght], '.', label=str(signals[i]))
    plt.legend()
    plt.title('outputs: ' + folder)
    plt.xlabel('frequency')
    plt.xscale("log")
    plt.ylabel('impedancia')
    if(plt_log_scale):
        plt.yscale("log")
    plt.rc('font', **font)
    plt.show()

#output everything (but jsut the 1st one)
    #Impedance
if(Impedance):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plts_file = int((signals_out[0].size)/signal_lenght)
    for plt_i in range(1):
        for i in range(num_signals):
            plt.plot(frequency, signals_out[i][plt_i*signal_lenght:(plt_i+1)*signal_lenght], '.', label=str(signals[i]))
    plt.legend()
    plt.title('impedance')
    plt.xlabel('frequency')
    plt.xscale("log")
    plt.ylabel('|impedance| [hom]')
    if(plt_log_scale):
        plt.yscale("log")
    plt.rc('font', **font)
    plt.show()

    #phase
if(phase):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plts_file = int((signals_out_phase[0].size)/signal_lenght)
    for plt_i in range(1):
        for i in range(num_signals):
            plt.plot(frequency, signals_out_phase[i][plt_i*signal_lenght:(plt_i+1)*signal_lenght], '.', label=str(signals[i]))
    plt.legend()
    plt.title('phase')
    plt.xlabel('frequency')
    plt.xscale("log")
    plt.ylabel('phase [rads]')
    if(plt_log_scale):
        plt.yscale("log")
    plt.rc('font', **font)
    plt.show()

    #Zr(frequency)
if(rImpedance):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    signal_real = []
    for i in range(num_signals):
        for j in range(signal_lenght):
            signal_real.append(signals_out[i][j]*math.cos(signals_out_phase[i][j]))
        ax1.plot(frequency, signal_real[:signal_lenght], '.', label=signals[i])
        ax2.plot(frequency, signal_real[:signal_lenght], '+', label=signals[i])
    ax1.legend(loc='center left')
    ax2.legend()
    plt.title('R{Z(f)}')
    plt.xlabel('frequency (Hz)')
    plt.xscale("log")
    plt.ylabel('impedancia (hom)')
    ax2.set_ylabel('log', color='tab:cyan')
    if(plt_log_scale):
        ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:cyan')
    plt.show()

    #Zi(frequency)
if(iImpedance):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    signal_real = []
    for i in range(num_signals):
        for j in range(signal_lenght):
            signal_real.append(signals_out[i][j]*math.sin(signals_out_phase[i][j]))
        ax1.plot(frequency, signal_real[:signal_lenght], '.', label=signals[i])
        ax2.plot(frequency, signal_real[:signal_lenght], '+', label=signals[i])
    ax1.legend(loc='center left')
    ax2.legend()
    plt.title('I{Z(f)}')
    plt.xlabel('frequency (Hz)')
    plt.xscale("log")
    plt.ylabel('impedancia (hom)')
    ax2.set_ylabel('log', color='tab:cyan')
    if(plt_log_scale):
        ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:cyan')
    plt.show()

#PCA
if(plt_PCAs):
    #calc
    x = StandardScaler().fit_transform(signals_out)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(concentration, columns=['target'])], axis = 1)
    #normalized data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plts_file = int((signals_out[0].size)/signal_lenght)
    for plt_i in range(1):
        for i in range(num_signals):
            plt.plot(frequency, x[i][plt_i*signal_lenght:(plt_i+1)*signal_lenght], '.', label=str(signals[i]))
    #plt pca scaled weights
    weights = pca.components_[0][:signal_lenght]
    weights = weights/abs(max(weights, key=abs))
    plt.plot(frequency, weights, label="scaled weights")
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
    plt.title('PCA calculations')
    plt.xlabel('frequency')
    plt.xscale("log")
    plt.ylabel('impedancia')
    plt.rc('font', **font)
    plt.show()
    #principale components
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    for target in concentration:
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , s = 50)
    ax.legend(signals)
    ax.grid()
    plt.show()

#HCA
if(plt_HCAs):
    import scipy.cluster.hierarchy as shc
    signals_out2 = []
    for i in range(num_signals):
        for plt_i in range(3):
            signals_out2.append(signals_out[i][plt_i*signal_lenght:(plt_i+1)*signal_lenght])
    signals2 = []
    for i in range(num_signals):
        for j in range(3):
            signals2.append(signals[i])
        #raw
    x = StandardScaler().fit_transform(signals_out2)
    xDf = pd.DataFrame(data = x)
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms raw")  
    dend = shc.dendrogram(shc.linkage(xDf, method='ward'), labels=signals2)
    plt.show()
        #PCAs
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms PCAs")  
    dend = shc.dendrogram(shc.linkage(principalDf, method='ward'), labels=signals)
    plt.show()
        #PCA1
    principalDf = pd.DataFrame(data = principalComponents[:,0], columns = ['principal component 1'])
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms PCA1")  
    dend = shc.dendrogram(shc.linkage(principalDf, method='ward'), labels=signals)
    plt.show()


