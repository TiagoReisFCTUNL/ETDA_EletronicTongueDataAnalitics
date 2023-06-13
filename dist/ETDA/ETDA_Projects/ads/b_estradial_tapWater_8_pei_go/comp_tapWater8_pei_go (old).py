# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:05:01 2022

@author: Asus
"""

#IMPORT
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from math import cos
from numpy import sin
from math import pi
from math import tan
import re

#DEF
def line_func(x, a, b):
    return a*x+b;

def invert(array):
    return array[::-1]

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#VALS
calibrate_with_control_sample = 0 #influences tendency curves: 0 -> ignores 0M; 1 -> uses 0M
control_samples = 1 #number of control samples 
num_samples = 3 #number of non-control samples 
first_concentration = 9 #min{|log(concentration)|}
skip=3  #interval between each log(concentration)




folder = os.path.basename(os.getcwd())
#define signals
tmp = os.listdir(os.getcwd())
    #identify folders
signals = []
for f in tmp:
    if f.find('.') == -1:
        #save environment_signals for each folder
        tmp2 = os.listdir(f)
        for file in tmp2:
            if file.find('.csv') != -1:
                tmp3 = file
                #filter temp3
                if tmp3.find('_') != -1:
                    tmp3 = tmp3[:tmp3.find('_')]
                else:
                    tmp3 = tmp3[:tmp3.find('.')]
                signals.append(tmp3)
        signals = list(dict.fromkeys(signals)) #remove duplicates (normal + corrected)
        signals = sorted_alphanumeric(signals)
        tmp4 = signals[1:]  #pass 0M to end of list
        tmp4.append(signals[0])
        signals = tmp4
        break

    #read files
samples = []
samples_phase = []
rsamples = []
isamples = []
tsamples = []

num_signals = len(signals)
c = np.arange(start=first_concentration, stop=first_concentration+(num_signals-1)*skip, step=skip)
c = np.concatenate((c,[18]))
x = np.arange(start=first_concentration, stop=first_concentration+(num_signals-1)*skip, step=0.1)

csv = (pd.read_csv(f"{folder}1\\{signals[0]}_corrected.csv", skiprows=2, sep=";")).to_numpy()
curr_signal_lenght = csv[:,1].tolist().count(1)

freq = csv[:curr_signal_lenght,4]
signal_lenght = curr_signal_lenght
total_samples = num_samples + control_samples
num_loops_file = int((csv[:,4].size)/signal_lenght)
for i in range(total_samples):
    csv_to_open = []    #len = signals
    if(i<num_samples):
        for s in signals:
            csv_to_open.append(f"{folder}{i+1}\\{s}_corrected.csv")
    else:
        for s in signals:
            csv_to_open.append(f"{folder[:folder[:folder.rfind('_')].rfind('_')]}\\{s}_corrected.csv")
    for l in range(num_loops_file):
        signals_out = []    #len = num_loops_file*signals
        signals_out_phase = []
        real_sample = []
        imaginary_sample = []
        tan_sample = []
        for j in range(num_signals):
            tmpr = []
            tmpi = []
            tmpt = []
            
            csv = (pd.read_csv(csv_to_open[j], skiprows=2, sep=";")).to_numpy()
            signals_out.append(csv[signal_lenght*l:signal_lenght*(l+1),10])
            signals_out_phase.append(csv[signal_lenght*l:signal_lenght*(l+1),11])
            
            for f in range(signal_lenght):
                signals_out_phase[j][f] = signals_out_phase[j][f]*(pi/180)
                tmpr.append(signals_out[j][f]*cos(signals_out_phase[j][f]))
                tmpi.append(signals_out[j][f]*sin(signals_out_phase[j][f]))
                tmpt.append(tan(signals_out_phase[j][f]))
            real_sample.append(tmpr)
            imaginary_sample.append(tmpi)
            tan_sample.append(tmpt)
        samples.append(signals_out)
        samples_phase.append(signals_out_phase)
        rsamples.append(real_sample)
        isamples.append(imaginary_sample)
        tsamples.append(tan_sample)

domains_labels = ['impedance', 'phase', 'real impedance', 'imaginary impedance', 'losses tangent']
ylables = ['|impedance| [ohm]', 'phase [rads]', 'R{Z} [ohm]', 'I{Z} [ohm]', 'tan(phase)']
domains = [samples, samples_phase, rsamples, isamples, tsamples]
num_domains = len(domains_labels)

def cal_curve_freq(y):
    popt, pcov = curve_fit(line_func, freq.copy()[1:], y.copy()[1:])
    return popt[0]*freq+popt[1]

def cal_curve(pc1):
    popt, pcov = curve_fit(line_func, c.copy()[:-control_samples*(1-calibrate_with_control_sample)], pc1.copy()[:-control_samples*(1-calibrate_with_control_sample)])
    return popt[0]*x+popt[1]

    #set const
total_lenght = samples[0][0].size
plts_file = int((samples[0][0].size)/signal_lenght)

simb = ['.', '+', '^', 's', '*', 'd', 'x', '-p']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

for d in range(num_domains):
        #normalize data
    samples_norm1 = []
    samples_norm2 = []
    
    samp1_mean = 0
    samp1_std = 0
    for i in range(total_samples):
        for l in range(num_loops_file):
            scaler = StandardScaler()
            scaler.fit(domains[d][i*num_loops_file+l].copy())
            norm_out = scaler.transform(domains[d][i*num_loops_file+l].copy())
            samples_norm1.append(norm_out)
            if(i==0):
                samp1_mean = samp1_mean + scaler.mean_
                samp1_std = samp1_std + (scaler.var_)**0.5
            #normalize data w 1st normalization
                if(l==num_loops_file-1):
                    samp1_mean = samp1_mean/3
                    samp1_std = samp1_std/3
                    for l2 in range(num_loops_file):
                        samples_norm2.append((domains[d][i*num_loops_file+l2] - samp1_mean)/samp1_std)
            else:
                samples_norm2.append((domains[d][i*num_loops_file+l] - samp1_mean)/samp1_std)
        #domain
    for i in range(num_samples):
        for j in range(num_signals):
            for l in range(num_loops_file):
                p = plt.plot(freq, domains[d][i*num_loops_file+l][j], simb[i], color=colors[j], markersize=5, label=f"{signals[j]}_s{i+1}_l{l+1}", alpha=2/(3+l))
    for i in range(control_samples):
        for j in range(num_signals):
            for l in range(num_loops_file):
                p = plt.plot(freq, domains[d][(num_samples+i)*num_loops_file+l][j], simb[num_samples+i], color=colors[j], markersize=5, label=f"{signals[j]}_control_l{l}", alpha=2/(3+l))
    plt.title(domains_labels[d])
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.ylabel(ylables[d])
    plt.legend(loc='upper left', ncol=int(num_signals/2), bbox_to_anchor=(1.04, 1.04))
    plt.show()
        #log domain
    if d == 0: #just for impedance (the others have negative values)
        for i in range(num_samples):
            for j in range(num_signals):
                for l in range(num_loops_file):
                    p = plt.plot(freq, domains[d][i*num_loops_file+l][j], simb[i], color=colors[j], markersize=5, label=f"{signals[j]}_s{i+1}_l{l+1}", alpha=2/(3+l))
        for i in range(control_samples):
            for j in range(num_signals):
                for l in range(num_loops_file):
                    p = plt.plot(freq, domains[d][(num_samples+i)*num_loops_file+l][j], simb[num_samples+i], color=colors[j], markersize=5, label=f"{signals[j]}_control_l{l}", alpha=2/(3+l))
        plt.title(domains_labels[d])
        plt.xlabel('frequency [Hz]')
        plt.xscale('log')
        plt.ylabel(f"log({ylables[d]})")
        plt.yscale('log')
        plt.legend(loc='upper left', ncol=int(num_signals/2), bbox_to_anchor=(1.04, 1.04))
        plt.show()
        #specter /M0
    for i in range(num_samples):
        for j in range(num_signals-1):
            for l in range(num_loops_file):
                p = plt.plot(freq, [a / b for a, b in zip(domains[d][i*num_loops_file+l][j], domains[d][i*num_loops_file+l][num_signals-1])], simb[i], color=colors[j], markersize=5, label=f"{signals[j]}_s{i+1}_l{l+1}", alpha=2/(3+l))
    for i in range(control_samples):
        for j in range(num_signals-1):
            for l in range(num_loops_file):
                p = plt.plot(freq, [a / b for a, b in zip(domains[d][(num_samples+i)*num_loops_file+l][j], domains[d][(num_samples+i)*num_loops_file+l][num_signals-1])], simb[num_samples+i], color=colors[j], markersize=5, label=f"{signals[j]}_control_l{l}", alpha=2/(3+l))
    plt.title(f"{domains_labels[d]} / M0")
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.ylabel(ylables[d])
    plt.legend(loc='upper left', ncol=int(num_signals/2), bbox_to_anchor=(1.04, 1.04))
    plt.show()
        #specter -M0
    for i in range(num_samples):
        for j in range(num_signals-1):
            for l in range(num_loops_file):
                p = plt.plot(freq, [a - b for a, b in zip(domains[d][i*num_loops_file+l][j], domains[d][i*num_loops_file+l][num_signals-1])], simb[i], color=colors[j], markersize=5, label=f"{signals[j]}_s{i+1}_l{l+1}", alpha=2/(3+l))
    for i in range(control_samples):
        for j in range(num_signals-1):
            for l in range(num_loops_file):
                p = plt.plot(freq, [a - b for a, b in zip(domains[d][(num_samples+i)*num_loops_file+l][j], domains[d][(num_samples+i)*num_loops_file+l][num_signals-1])], simb[num_samples+i], color=colors[j], markersize=5, label=f"{signals[j]}_control_l{l}", alpha=2/(3+l))
    plt.title(f"{domains_labels[d]} - M0")
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.ylabel(ylables[d])
    plt.legend(loc='upper left', ncol=int(num_signals/2), bbox_to_anchor=(1.04, 1.04))
    plt.show()
        #domain 1Hz
    avg = np.linspace(0,0,num_signals)
    avg1 = np.linspace(0,0,num_signals)
    for i in range(num_samples):
        for l in range(num_loops_file):
            tmp = []
            for j in range(num_signals):
                tmp.append(domains[d][i*num_loops_file+l][j][signal_lenght-1])
            p = plt.plot(-c, tmp, simb[i], color=colors[i], markersize=5, label=f"sample {i+1} loop {l+1}", alpha=2/(3+l))
            plt.plot(-x, cal_curve(tmp), color=colors[i], alpha=2/(3+l))
            avg = np.add(avg,tmp)
            if l == 0:
                avg1 = np.add(avg1,tmp)
    for i in range(control_samples):
        for l in range(num_loops_file):
            tmp = []
            for j in range(num_signals):
                tmp.append(domains[d][(num_samples+i)*num_loops_file+l][j][signal_lenght-1])
            p = plt.plot(-c, tmp, simb[num_samples+i], color=colors[num_samples+i], markersize=5, label=f"control {i+1} loop {l+1}", alpha=2/(3+l))
            plt.plot(-x, cal_curve(tmp), color=colors[num_samples+i])
    for i in range(len(avg)):
        avg[i]=avg[i]/(num_samples*num_loops_file)
        avg1[i]=avg1[i]/(num_samples)
    plt.plot(-c, avg1, simb[total_samples], color=colors[total_samples], markersize=10, label='average loop 1', alpha=3/4)
    plt.plot(-x,cal_curve(avg1), color=colors[total_samples])
    plt.plot(-c, avg, simb[total_samples+1], color=colors[total_samples+1], markersize=8, label='average', alpha=3/4)
    plt.plot(-x,cal_curve(avg), color=colors[total_samples+1])
    plt.title(f"{domains_labels[d]}(1Hz)")
    plt.xlabel('log(concentration)')
    plt.ylabel(ylables[d])
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
    plt.show()
        #apply difrent weights and standards to data set
            #normal PCA1
    avg = np.linspace(0,0,num_signals)
    avg1 = np.linspace(0,0,num_signals)
    weights = 0
    for i in range(num_samples):
        for l in range(num_loops_file):
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(samples_norm1[i*num_loops_file+l])
                    #plot points
            p = plt.plot(-c, principalComponents[:,0], simb[i], color=colors[i], markersize=5, label=f"sample {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents[:,0]), color=colors[i])
                    #save 1st weights
            if(i == 0 ):
                if(l == 0):
                    weights = pca.components_[0]
                else:
                    weights = weights + pca.components_[0]
                    #add principalComponents to avgs
            avg = np.add(avg,principalComponents[:,0])
            if l == 0:
                avg1 = np.add(avg1,principalComponents[:,0])
                #avg weights of 1st sample
        if(i == 0):
            for w in range(len(weights)):
                weights[w] = weights[w]/3
    for i in range(control_samples):
        for l in range(num_loops_file):
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(samples_norm1[(num_samples+i)*num_loops_file+l])
                    #plot points
            p = plt.plot(-c, principalComponents[:,0], simb[num_samples+i], color=colors[num_samples+i], markersize=5, label=f"control {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents[:,0]), color=colors[num_samples+i])
    for i in range(len(avg)):
        avg[i]=avg[i]/(num_samples*num_loops_file)
        avg1[i]=avg1[i]/(num_samples)
    plt.plot(-c, avg1, simb[total_samples], color=colors[total_samples], markersize=10, label='average loop 1', alpha=3/4)
    plt.plot(-x,cal_curve(avg1), color=colors[total_samples])
    plt.plot(-c, avg, simb[total_samples+1], color=colors[total_samples+1], markersize=8, label='average', alpha=3/4)
    plt.plot(-x,cal_curve(avg), color=colors[total_samples+1])
    plt.title('individual normalization and weights')
    plt.xlabel('log(concentration)')
    plt.ylabel('PC1(' + domains_labels[d] + ')')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
    plt.show()
            #PCA1 w 1st weights
    avg = np.linspace(0,0,num_signals)
    avg1 = np.linspace(0,0,num_signals)
    for i in range(num_samples):
        for l in range(num_loops_file):
            principalComponents = []
            for j in range(len(samples_norm1[i*num_loops_file+l])):
                principalComponents.append(np.dot(samples_norm1[i*num_loops_file+l][j],weights))
                    #plot
            p = plt.plot(-c, principalComponents, simb[i], color=colors[i], markersize=5, label=f"sample {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents), color=colors[i])
                    #add principalComponents to avgs
            avg = np.add(avg,principalComponents)
            if l == 0:
                avg1 = np.add(avg1,principalComponents)
    for i in range(control_samples):
        for l in range(num_loops_file):
            principalComponents = []
            for j in range(len(samples_norm1[(num_samples+i)*num_loops_file+l])):
                principalComponents.append(np.dot(samples_norm1[(num_samples+i)*num_loops_file+l][j],weights))
                    #plot
            p = plt.plot(-c, principalComponents, simb[num_samples+i], color=colors[num_samples+i], markersize=5, label=f"control {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents), color=colors[num_samples+i])
    for i in range(len(avg)):
        avg[i]=avg[i]/(num_samples*num_loops_file)
        avg1[i]=avg1[i]/(num_samples)
    plt.plot(-c, avg1, simb[total_samples], color=colors[total_samples], markersize=10, label='average loop 1', alpha=3/4)
    plt.plot(-x,cal_curve(avg1), color=colors[total_samples])
    plt.plot(-c, avg, simb[total_samples+1], color=colors[total_samples+1], markersize=8, label='average', alpha=3/4)
    plt.plot(-x,cal_curve(avg), color=colors[total_samples+1])
    plt.title('individual normalization, shared weights')
    plt.xlabel('log(concentration)')
    plt.ylabel('PC1(' + domains_labels[d] + ')')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
    plt.show()
            #normal PCA1 w 1st normalization
    avg = np.linspace(0,0,num_signals)
    avg1 = np.linspace(0,0,num_signals)
    for i in range(num_samples):
        for l in range(num_loops_file):
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(samples_norm2[i*num_loops_file+l])
                    #plot
            p = plt.plot(-c, principalComponents[:,0], simb[i], color=colors[i], markersize=5, label=f"sample {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents[:,0]), color=colors[i])
                    #add principalComponents to avgs
            avg = np.add(avg,principalComponents[:,0])
            if l == 0:
                avg1 = np.add(avg1,principalComponents[:,0])
    for i in range(control_samples):
        for l in range(num_loops_file):
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(samples_norm2[(num_samples+i)*num_loops_file+l])
                    #plot
            p = plt.plot(-c, principalComponents[:,0], simb[num_samples+i], color=colors[num_samples+i], markersize=5, label=f"control {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents[:,0]), color=colors[num_samples+i])
    for i in range(len(avg)):
        avg[i]=avg[i]/(num_samples*num_loops_file)
        avg1[i]=avg1[i]/(num_samples)
    plt.plot(-c, avg1, simb[total_samples], color=colors[total_samples], markersize=10, label='average loop 1', alpha=3/4)
    plt.plot(-x,cal_curve(avg1), color=colors[total_samples])
    plt.plot(-c, avg, simb[total_samples+1], color=colors[total_samples+1], markersize=8, label='average', alpha=3/4)
    plt.plot(-x,cal_curve(avg), color=colors[total_samples+1])
    plt.title('shared normalization, individual weights')
    plt.xlabel('log(concentration)')
    plt.ylabel('PC1(' + domains_labels[d] + ')')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
    plt.show()
            #PCA1 with 1st weight + normalization
    avg = np.linspace(0,0,num_signals)
    avg1 = np.linspace(0,0,num_signals)
    for i in range(num_samples):
        for l in range(num_loops_file):
            principalComponents = []
            for j in range(len(samples_norm2[i*num_loops_file+l])):
                principalComponents.append(np.dot(samples_norm2[i*num_loops_file+l][j],weights))
                    #plot
            p = plt.plot(-c, principalComponents, simb[i], color=colors[i], markersize=5, label=f"sample {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents), color=colors[i])
                    #add principalComponents to avgs
            avg = np.add(avg,principalComponents)
            if l == 0:
                avg1 = np.add(avg1,principalComponents)
    for i in range(control_samples):
        for l in range(num_loops_file):
            principalComponents = []
            for j in range(len(samples_norm2[(num_samples+i)*num_loops_file+l])):
                principalComponents.append(np.dot(samples_norm2[(num_samples+i)*num_loops_file+l][j],weights))
                    #plot
            p = plt.plot(-c, principalComponents, simb[num_samples+i], color=colors[num_samples+i], markersize=5, label=f"control {i+1} loop {l+1}", alpha=2/(3+l))
                    #plot tend curve
            plt.plot(-x, cal_curve(principalComponents), color=colors[num_samples+i])
    for i in range(len(avg)):
        avg[i]=avg[i]/(num_samples*num_loops_file)
        avg1[i]=avg1[i]/(num_samples)
    plt.plot(-c, avg1, simb[total_samples], color=colors[total_samples], markersize=10, label='average loop 1', alpha=3/4)
    plt.plot(-x,cal_curve(avg1), color=colors[total_samples])
    plt.plot(-c, avg, simb[total_samples+1], color=colors[total_samples+1], markersize=8, label='average', alpha=3/4)
    plt.plot(-x,cal_curve(avg), color=colors[total_samples+1])
    plt.title('shared normalization and weights')
    plt.xlabel('log(concentration)')
    plt.ylabel('PC1(' + domains_labels[d] + ')')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
    plt.show()





#Nyquist------------------------------------------
    #apply difrent weights and standards to data set
        #n normalization
color=['tab:blue', 'tab:orange', 'tab:green']
for i in range(num_samples):
    for j in range(len(isamples[0])):
        plt.plot(rsamples[i][j][1:8], isamples[i][j][1:8], simb[i], markersize=10/(i+1), color=color[i])
        plt.plot(rsamples[i][j][1:8], isamples[i][j][1:8], color=color[i])
plt.title("Nyquist")
plt.ylabel("I{Z} [ohm]")
plt.xlabel("R{Z} [ohm]")
plt.show()
nsamples = []
for i in range(num_samples):
    index = []
    for j in range(len(isamples[0])):
            #plt
        plt.plot(rsamples[i][j][1:8], isamples[i][j][1:8], label=signals[j])
        plt.plot(rsamples[i][j][1:8], isamples[i][j][1:8], simb[i], markersize=10/(i+1), color=color[i])
            #get max
        index.append(rsamples[i][j][isamples[i][j][1:8].index(max(isamples[i][j][1:8]))])
            #plt Nyquist
    plt.title("Nyquist sample" + str(i+1))
    plt.ylabel("I{Z} [ohm]")
    plt.xlabel("R{Z} [ohm]")
    plt.legend(bbox_to_anchor=(1, 1.05))
    plt.show()
            #plt max
    plt.plot(-c, index, color=color[i])
    plt.plot(-c, index, simb[i], markersize=10/(i+1), color=color[i])
    plt.title("Nyquist sample" + str(i+1))
    plt.ylabel("R{Z} [ohm]")
    plt.xlabel("log(concentration)")
    plt.show()
    nsamples.append(index)
            #plt all max
for i, niq in enumerate(nsamples):
    plt.plot(-c, niq, color=color[i])
    plt.plot(-c, niq, simb[i], markersize=10/(i+1), color=color[i])
plt.title("Nyquist all sample")
plt.ylabel("R{Z} [ohm]")
plt.xlabel("log(concentration)")
plt.show()





#loss tg 100Hz-1Hz
avg = []
for i in range(num_samples):
    for l in range(num_loops_file):
        tmp = []
        for j in range(len(tsamples[i*num_loops_file+l])):
            tmp.append(tsamples[i*num_loops_file+l][j][freq.tolist().index(1)]-tsamples[i*num_loops_file+l][j][freq.tolist().index(100)])
        plt.plot(-c,tmp, simb[i], color=colors[i], markersize=5, label='sample '+str(i+1)+'_l'+str(l+1), alpha=2/(3+l))
        if(i==0 and l==0):
            avg=tmp
        else:
            avg = np.add(avg,tmp)
for i in range(len(avg)):
    avg[i]=avg[i]/(num_samples*num_loops_file)
plt.plot(-c, avg, simb[num_samples], color=colors[num_samples], markersize=10, label='avg')
plt.plot(-x,cal_curve(avg), color=colors[num_samples])
plt.title('loss tg 100Hz-1Hz')
plt.xlabel('log(concentration)')
plt.ylabel('PC1(' + domains_labels[d] + ')')
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
plt.show()