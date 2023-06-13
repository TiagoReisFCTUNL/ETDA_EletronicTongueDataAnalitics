# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:46:25 2022

@author: Asus
"""

# IMPORT
import math
import os
import sklearn
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from math import pi, tan


class Sample:
    # DEF
    @staticmethod
    def line_func(x, a, b):
        return a * x + b

    @staticmethod
    def invert(array):
        return array[::-1]

    def set_plt_log_scale(self, value):
        if type(value) is bool:
            self.plt_log_scale = value

    def __init__(self, plt_log_scale=False, font=None):
        # init empty vars
        self.path = None
        self.frequency = None
        self.signals_out_phase = None
        self.signals_out = None
        self.plots_file = None
        self.total_length = None
        self.signal_length = None
        self.num_signals = None
        self.folder = None
        self.csv_to_open = None
        self.signals = None
        self.filenames = None
        # init vars with values
        if font and type(font) is dict:
            self.font = font
        else:
            self.font = {'size': 8}
        self.plt_log_scale = plt_log_scale

    def read_files(self, path):
        # get filenames
        self.path = path
        self.folder = os.path.basename(path)
        self.filenames = os.listdir(path)
        temp = []
        for f in self.filenames:
            if f[f.find('.'):] == ".csv":
                temp.append(f)
        self.csv_to_open = []
        self.signals = []
        self.filenames = [file for file in self.filenames if '.csv' in file]
        for s in self.filenames:
            if s[-14:] == '_corrected.csv':
                temp.remove(s)
            else:
                if s.find('_') != -1:
                    curr_signal = s[:s.find('_')]
                else:
                    curr_signal = s[:-4]
                self.signals.append(curr_signal)
                self.csv_to_open.append(curr_signal + '_corrected.csv')
        self.filenames = temp

        # ready csv files
        columns = "Result Number;Sweep Number;Point Number;Time;frequency (Hz);AC Level (V);DC Level (V);Set Point ('C);Temperature ('C);Control ('C);Impedance Magnitude (Ohms);Impedance Phase Degrees (');Admittance Magnitude (S);Capacitance Magnitude (F);"
        for i, file in enumerate(self.filenames):
            with open(join(path, self.csv_to_open[i]), 'w') as f_corrected:
                with open(join(path, file)) as f:
                    contents = f.readlines()
                doc_len = len(contents)
                contents[2] = columns
                for j in range(3):
                    f_corrected.writelines(contents[j])
                for j in range(doc_len - 3):
                    f_corrected.writelines(contents[j + 3].replace(',', '.'))

        # get values from signals
        self.num_signals = len(self.csv_to_open)
        self.signals_out = []
        self.signals_out_phase = []
        csv = None
        for i in range(self.num_signals):
            csv = (pd.read_csv(join(path, self.csv_to_open[i]), skiprows=2, sep=";")).to_numpy()
            self.signals_out.append(csv[:, 10])
            self.signals_out_phase.append(csv[:, 11])
        self.signal_length = csv[:, 1].tolist().count(1)
        self.frequency = csv[:self.signal_length, 4]

        self.total_length = self.signals_out[0].size
        self.plots_file = int(self.signals_out[0].size / self.signal_length)

        # invert
        self.frequency = Sample.invert(self.frequency)
        for i in range(self.num_signals):
            self.signals_out[i] = Sample.invert(self.signals_out[i])
            self.signals_out_phase[i] = Sample.invert(self.signals_out_phase[i])

        # convert degrees to rads
        for i in range(self.num_signals):
            for f in range(self.signal_length):
                self.signals_out_phase[i][f] = self.signals_out_phase[i][f] * (pi / 180)

    def plt_each_signal(self):  # creates an Impedance(frequency) plt for each concentration
        plt.rc('font', **self.font)
        fig = plt.figure()
        fig.add_subplot(111)
        for i in range(self.num_signals):
            plt.clf()
            plt.figure(figsize=(8., 8.))
            for plt_i in range(self.plots_file):
                plt.plot(self.frequency,
                         self.signals_out[i][plt_i * self.signal_length:(plt_i + 1) * self.signal_length], '.',
                         label=str(self.signals[i]))
            plt.legend()
            plt.title('impedance')
            plt.xlabel('frequency')
            plt.xscale("log")
            plt.ylabel('|impedance| [hom]')
            if self.plt_log_scale:
                plt.yscale("log")
            # plt.show()

    def plt_all(self):  # creates an Impedance(frequency) plt with all the concentrations and loops
        plt.rc('font', **self.font)
        fig = plt.figure()
        fig.add_subplot(111)
        for plt_i in range(self.plots_file):
            plt.clf()
            plt.figure(figsize=(8., 8.))
            for i in range(self.num_signals):
                plt.plot(self.frequency,
                         self.signals_out[i][plt_i * self.signal_length:(plt_i + 1) * self.signal_length], '.',
                         label=str(self.signals[i]))
        plt.legend()
        plt.title('outputs: ' + self.folder)
        plt.xlabel('frequency')
        plt.xscale("log")
        plt.ylabel('impedance')
        if self.plt_log_scale:
            plt.yscale("log")
        # plt.show()

    def results(self, impedance=False, phase=False, r_impedance=False, i_impedance=False):
        imgs = []
        plt.rc('font', **self.font)

        # Impedance
        if impedance:
            fig = plt.figure()
            fig.add_subplot(111)
            plt.clf()
            plt.figure(figsize=(8., 6.))
            for plt_i in range(1):
                for i in range(self.num_signals):
                    plt.plot(self.frequency,
                             self.signals_out[i][plt_i * self.signal_length:(plt_i + 1) * self.signal_length], '.',
                             label=str(self.signals[i]))
            lgd = plt.legend()
            plt.title('impedance')
            plt.xlabel('frequency')
            plt.xscale("log")
            plt.ylabel('|impedance| [hom]')
            if self.plt_log_scale:
                plt.yscale("log")
            title = f"impedance.png"
            plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
            imgs.append(join(self.path, title))

        # phase
        if phase:
            fig = plt.figure()
            fig.add_subplot(111)
            plt.clf()
            plt.figure(figsize=(8., 6.))
            for plt_i in range(1):
                for i in range(self.num_signals):
                    plt.plot(self.frequency,
                             self.signals_out_phase[i][plt_i * self.signal_length:(plt_i + 1) * self.signal_length],
                             '.',
                             label=str(self.signals[i]))
            lgd = plt.legend()
            plt.title('phase')
            plt.xlabel('frequency')
            plt.xscale("log")
            plt.ylabel('phase [rads]')
            title = f"phase.png"
            plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
            imgs.append(join(self.path, title))

        # Zr(frequency)
        if r_impedance:
            plt.clf()
            fig, ax1 = plt.subplots(figsize=(8., 6.))
            all_signal_real = []
            for i in range(self.num_signals):
                signal_real = []
                for j in range(self.signal_length):
                    signal_real.append(self.signals_out[i][j] * math.cos(self.signals_out_phase[i][j]))
                ax1.plot(self.frequency, signal_real[:self.signal_length], '.', label=self.signals[i])
                all_signal_real.extend(signal_real)
            ax1.legend(loc='center left')
            plt.title('R{Z(f)}')
            plt.xlabel('frequency (Hz)')
            plt.xscale("log")
            plt.ylabel('impedance (hom)')
            if self.plt_log_scale:
                if all([value > 0 for value in all_signal_real]):
                    ax2 = ax1.twinx()
                    for i in range(self.num_signals):
                        signal_real = []
                        for j in range(self.signal_length):
                            signal_real.append(self.signals_out[i][j] * math.cos(self.signals_out_phase[i][j]))
                        ax2.plot(self.frequency, signal_real[:self.signal_length], '+', label=self.signals[i])
                    ax2.legend()
                    ax2.set_ylabel('log', color='tab:cyan')
                    ax2.set_yscale('log')
                    ax2.tick_params(axis='y', labelcolor='tab:cyan')
            lgd = plt.legend()
            title = f"real_impedance.png"
            plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
            imgs.append(join(self.path, title))

        # Zi(frequency)
        if i_impedance:
            fig, ax1 = plt.subplots(figsize=(8., 6.))
            all_signal_imaginary = []
            for i in range(self.num_signals):
                signal_imaginary = []
                for j in range(self.signal_length):
                    signal_imaginary.append(self.signals_out[i][j] * math.sin(self.signals_out_phase[i][j]))
                ax1.plot(self.frequency, signal_imaginary[:self.signal_length], '.', label=self.signals[i])
                all_signal_imaginary.extend(signal_imaginary)
            ax1.legend(loc='center left')
            plt.title('I{Z(f)}')
            plt.xlabel('frequency (Hz)')
            plt.xscale("log")
            plt.ylabel('impedance (hom)')
            if self.plt_log_scale:
                if all([value > 0 for value in all_signal_imaginary]):
                    ax2 = ax1.twinx()
                    for i in range(self.num_signals):
                        signal_imaginary = []
                        for j in range(self.signal_length):
                            signal_imaginary.append(self.signals_out[i][j] * math.sin(self.signals_out_phase[i][j]))
                        ax2.plot(self.frequency, signal_imaginary[:self.signal_length], '+', label=self.signals[i])
                    ax2.legend()
                    ax2.set_ylabel('log', color='tab:cyan')
                    ax2.set_yscale('log')
                    ax2.tick_params(axis='y', labelcolor='tab:cyan')
            lgd = plt.legend()
            title = f"imaginary_impedance.png"
            plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
            imgs.append(join(self.path, title))

        # Tan(frequency)
        if i_impedance:
            fig, ax1 = plt.subplots(figsize=(8., 6.))
            for i in range(self.num_signals):
                signal_tan = []
                for j in range(self.signal_length):
                    signal_tan.append(self.signals_out[i][j] * tan(self.signals_out_phase[i][j]))
                ax1.plot(self.frequency, signal_tan[:self.signal_length], '.', label=self.signals[i])
            ax1.legend(loc='center left')
            plt.title('tan(phase(f))')
            plt.xlabel('frequency (Hz)')
            plt.xscale("log")
            plt.ylabel('loss tangent')
            lgd = plt.legend()
            title = f"tan_impedance.png"
            plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
            imgs.append(join(self.path, title))
        return imgs

    # PCA
    def plt_pca(self, first_concentration, skip):  # creates 2 plots with PCA of the files in folders
        # calc
        concentration = np.arange(start=first_concentration,
                                  stop=first_concentration + self.num_signals,
                                  step=skip)
        x = StandardScaler().fit_transform(self.signals_out)
        pca = sklearn.decomposition.PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
        final_df = pd.concat([principal_df, pd.DataFrame(concentration, columns=['target'])], axis=1)
        # normalized data
        fig = plt.figure()
        fig.add_subplot(111)
        for plt_i in range(1):
            for i in range(self.num_signals):
                plt.plot(self.frequency, x[i][plt_i * self.signal_length:(plt_i + 1) * self.signal_length], '.',
                         label=str(self.signals[i]))
        # plt pca scaled weights
        weights = pca.components_[0][:self.signal_length]
        weights = weights / abs(max(weights, key=abs))
        plt.plot(self.frequency, weights, label="scaled weights")
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
        plt.title('PCA calculations')
        plt.xlabel('frequency')
        plt.xscale("log")
        plt.ylabel('impedance')
        plt.rc('font', **self.font)
        plt.show()
        # principal components
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        for target in concentration:
            indices_to_keep = final_df['target'] == target
            ax.scatter(final_df.loc[indices_to_keep, 'principal component 1'],
                       final_df.loc[indices_to_keep, 'principal component 2'],
                       s=50)
        ax.legend(self.signals)
        ax.grid()
        plt.show()

    # HCA
    def plt_hca(self):  # creates 3 plots with HCA of the files in folders
        x = StandardScaler().fit_transform(self.signals_out)
        pca = sklearn.decomposition.PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        signals_out2 = []
        for i in range(self.num_signals):
            for plt_i in range(3):
                signals_out2.append(self.signals_out[i][plt_i * self.signal_length:(plt_i + 1) * self.signal_length])
        signals2 = []
        for i in range(self.num_signals):
            for j in range(3):
                signals2.append(self.signals[i])
            # raw
        signals2 = np.array(signals2)
        x = StandardScaler().fit_transform(signals_out2)
        x_df = pd.DataFrame(data=x).to_numpy()
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram raw")
        shc.dendrogram(shc.linkage(x_df, method='ward'), labels=signals2)
        plt.show()
        # PCAs
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram PCAs")
        principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2']).to_numpy()
        shc.dendrogram(shc.linkage(principal_df, method='ward'), labels=self.signals)
        plt.show()
        # PCA1
        principal_df = pd.DataFrame(data=principal_components[:, 0], columns=['principal component 1']).to_numpy()
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram PCA1")
        shc.dendrogram(shc.linkage(principal_df, method='ward'), labels=self.signals)
        plt.show()
