# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:05:01 2022

@author: Asus
"""

# IMPORT
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from os import walk
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from math import cos
from numpy import sin
from math import pi
from math import tan
from scipy.spatial.distance import euclidean
import re
import warnings
from os.path import join


class Experiment:
    # DEF
    @staticmethod
    def __line_func__(x, a, b):
        return a * x + b

    @staticmethod
    def __invert_list__(array):
        return array[::-1]

    @staticmethod
    def __convert(text):
        return int(text) if text.isdigit() else text.lower()

    @staticmethod
    def __alphanum_key(key):
        return [Experiment.__convert(c) for c in re.split('([0-9]+)', key)]

    @staticmethod
    def __sorted_alphanumeric__(data):
        return sorted(data, key=Experiment.__alphanum_key)

    @staticmethod
    def __float_to_units__(f):
        q = f
        units = 0
        if f > 1:
            while units < 3:
                q = int(q / 1000)
                if q == 0:
                    break
                units += 1
            unit_simbols = [" ", " K", " M", " G"]
            return str(round(f * 10 ** (-units * 3), 2)) + unit_simbols[units]
        else:
            while units < 3:
                if q >= 1:
                    break
                q = q * 1000
                units += 1
            unit_simbols = [" ", " m", " u", " n"]
            return str(round(f * 10 ** (units * 3), 2)) + unit_simbols[units]

    def __cal_curve__(self, pc1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            concentrations = self.c.copy()[:-1] if self.calibrate_with_control_sample == 0 else self.c.copy()
            vec_in = pc1.copy()[:-1] if self.calibrate_with_control_sample == 0 else pc1.copy()
            popt, pcov, *_ = curve_fit(Experiment.__line_func__, concentrations, vec_in)
            length = len(vec_in)
            outliers_index = []
            if length > 3:
                res = popt[0] * concentrations + popt[1]
                base_deviation = euclidean(vec_in, res)
                for i in range(length):
                    concentrations_tmp = np.array([concentrations[j] for j in range(length) if j != i]).astype(int)
                    vec_in_tmp = [vec_in[j] for j in range(length) if j != i]
                    popt, pcov, *_ = curve_fit(Experiment.__line_func__, concentrations_tmp, vec_in_tmp)
                    res_tmp = popt[0] * concentrations_tmp + popt[1]
                    curr_deviation = euclidean(vec_in_tmp, res_tmp)
                    # outlier detection:
                    # curr_d/(length-1)^2 < base_d/length^2 x (1 - 1/length)
                    # (=) curr_d < base_d x ((length-1)/length)^2 x (1 - 1/length)
                    # (=) curr_d < base_d x (1 - 1/length)^3
                    if curr_deviation < ((1 - 1. / length) ** 3) * base_deviation:
                        outliers_index.append(i)
                concentrations = [concentrations[j] for j in range(length) if j not in outliers_index]
                vec_in = [vec_in[j] for j in range(length) if j not in outliers_index]
                popt, pcov, *_ = curve_fit(Experiment.__line_func__, concentrations, vec_in)
            return popt[0] * self.x + popt[1]

    def __init__(self, samples, first_concentration, skip, path, calibrate_with_control_sample=False):
        # init null vars
        self.num_samples = None
        self.tsamples = None
        self.rsamples = None
        self.isamples = None
        self.handles = None
        self.num_loops_file = None
        self.total_samples = None
        self.signal_length = None
        self.log_c = None
        self.log_freq = None
        self.plts_file = None
        self.total_length = None
        self.num_domains = None
        self.domains = None
        self.freq = None

        # save init parameters
        self.calibrate_with_control_sample = calibrate_with_control_sample  # influences tendency curves: 0 -> ignores 0M; 1 -> uses 0M
        self.control_samples = samples[0]  # list of control samples
        self.experiment_samples = samples[1]  # list of experiment samples
        self.path = path

        # set const
        self.simb = ['o', 'P', '^', 's', '*', 'd', 'x', 'p']
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                       'tab:gray']
        self.color_maps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Greys']
        self.font = {'size': 12}
        self.domains_labels = ['amplitude', 'phase', 'real impedance', 'imaginary impedance', 'losses tangent']
        self.image_labels = ['|impedance| [ohm]', 'phase [rads]', 'R{Z} [ohm]', 'I{Z} [ohm]', 'tan(phase)']

        # identify folders
        self.folder = os.path.basename(path)

        # define signals
        signals = []
        tmp = os.listdir(path)
        for f in tmp:
            if f.find('.') == -1:
                # save environment_signals for each folder
                tmp2 = os.listdir(join(path, f))
                for file in tmp2:
                    if file.find('.csv') != -1:
                        tmp3 = file
                        # filter temp3
                        if tmp3.find('_') != -1:
                            tmp3 = tmp3[:tmp3.find('_')]
                        else:
                            tmp3 = tmp3[:tmp3.find('.')]
                        signals.append(tmp3)
                signals = list(dict.fromkeys(signals))  # remove duplicates (normal + corrected)
                signals = Experiment.__sorted_alphanumeric__(signals)
                tmp4 = signals[1:]  # pass 0M to end of list
                tmp4.append(signals[0])
                signals = tmp4
                break
        self.signals = signals
        num_signals = len(signals)
        self.num_signals = num_signals
        c = np.arange(start=first_concentration, stop=first_concentration + (num_signals - 1) * skip,
                      step=skip)
        control_sample = 18 if first_concentration + (num_signals-1) * skip < 18\
            else first_concentration + (num_signals-1) * skip
        c = np.concatenate((c, [control_sample]))
        self.c = c
        self.x = np.arange(start=first_concentration, stop=first_concentration + (num_signals - 1) * skip, step=0.1)

    def get_values(self):
        # read files
        samples = []
        samples_phase = []
        rsamples = []
        isamples = []
        tsamples = []

        csv = (pd.read_csv(join(self.path, self.experiment_samples[0], self.signals[0] + "_corrected.csv"), skiprows=2,
                           sep=";")).to_numpy()
        curr_signal_length = csv[:, 1].tolist().count(1)

        self.freq = csv[:curr_signal_length, 4]
        signal_length = curr_signal_length
        # total_samples = self.num_samples + len(self.control_samples)
        total_samples = len(next(walk(self.path))[1])
        self.num_samples = total_samples - len(self.control_samples)
        num_loops_file = int(csv[:, 4].size / signal_length)
        self.signal_length = signal_length
        self.total_samples = total_samples
        self.num_loops_file = num_loops_file
        csv_to_open = []  # len = signals
        for sample in self.experiment_samples:
            csv_from_sample = []
            for s in self.signals:
                csv_from_sample.append(join(self.path, sample, s + "_corrected.csv"))
            csv_to_open.append(csv_from_sample)
        for sample in self.control_samples:
            csv_from_sample = []
            for s in self.signals:
                csv_from_sample.append(join(self.path, sample, s + "_corrected.csv"))
            csv_to_open.append(csv_from_sample)
        for i in range(total_samples):
            for loop in range(num_loops_file):
                signals_out = []  # len = num_loops_file*signals
                signals_out_phase = []
                real_sample = []
                imaginary_sample = []
                tan_sample = []
                for j in range(self.num_signals):
                    tmpr = []
                    tmpi = []
                    tmpt = []
                    csv = (pd.read_csv(csv_to_open[i][j], skiprows=2, sep=";")).to_numpy()
                    signals_out.append(csv[signal_length * loop:signal_length * (loop + 1), 10])
                    signals_out_phase.append(csv[signal_length * loop:signal_length * (loop + 1), 11])

                    for f in range(signal_length):
                        signals_out_phase[j][f] = signals_out_phase[j][f] * (pi / 180)
                        tmpr.append(signals_out[j][f] * cos(signals_out_phase[j][f]))
                        tmpi.append(signals_out[j][f] * sin(signals_out_phase[j][f]))
                        tmpt.append(tan(signals_out_phase[j][f]))
                    real_sample.append(tmpr)
                    imaginary_sample.append(tmpi)
                    tan_sample.append(tmpt)
                samples.append(signals_out)
                samples_phase.append(signals_out_phase)
                rsamples.append(real_sample)
                isamples.append(imaginary_sample)
                tsamples.append(tan_sample)
        self.isamples = isamples
        self.rsamples = rsamples
        self.tsamples = tsamples

        self.domains = [samples, samples_phase, rsamples, isamples, tsamples]
        self.num_domains = len(self.domains_labels)
        self.log_freq = np.log10(self.freq.astype(float))
        self.log_c = []
        for j in range(self.num_signals):
            self.log_c.append([self.c[j]] * signal_length)
        self.handles = [mlines.Line2D([], [], color=self.colors[i], marker=self.simb[i], linestyle='None', markersize=5,
                                      label=f"sample {i + 1}" if i < self.num_samples else f"control {i - self.num_samples + 1}")
                        for i in
                        range(total_samples)]
        # set const
        self.total_length = samples[0][0].size
        self.plts_file = int(samples[0][0].size / signal_length)

    def show_results(self, show_all=False, show_domains=False, show_db=False, show_max_variance=False,
                     show_normal_pc1=False, show_pc1_sample1_weight=False, show_pc1_sample1_normalization=False,
                     show_pc1_sample1_weight_normalization=False):
        # resolve show_all
        if (not (show_domains or show_db or show_max_variance or show_normal_pc1 or show_pc1_sample1_weight or
                 show_pc1_sample1_normalization or show_pc1_sample1_weight_normalization)) or show_all:
            show_domains = True
            show_db = True
            show_max_variance = True
            show_normal_pc1 = True
            show_pc1_sample1_weight = True
            show_pc1_sample1_normalization = True
            show_pc1_sample1_weight_normalization = True

        # create local vars
        imgs = []
        total_samples = self.total_samples
        num_loops_file = self.num_loops_file
        num_samples = self.num_samples
        num_signals = self.num_signals
        control_samples = total_samples - num_samples
        plt.rc('font', **self.font)
        plt.rc('figure', max_open_warning=0)
        for d in range(self.num_domains):
            # normalize data
            samples_norm1 = []
            samples_norm2 = []
            samp1_mean = 0
            samp1_std = 0
            for i in range(total_samples):
                for loop in range(num_loops_file):
                    scaler = StandardScaler()
                    scaler.fit(self.domains[d][i * num_loops_file + loop].copy())
                    norm_out = scaler.transform(self.domains[d][i * num_loops_file + loop].copy())
                    samples_norm1.append(norm_out)
                    if i == 0:
                        samp1_mean = samp1_mean + scaler.mean_
                        samp1_std = samp1_std + scaler.var_ ** 0.5
                        # normalize data w 1st normalization
                        if loop == num_loops_file - 1:
                            samp1_mean = samp1_mean / num_loops_file
                            samp1_std = samp1_std / num_loops_file
                            for l2 in range(num_loops_file):
                                samples_norm2.append(
                                    (self.domains[d][i * num_loops_file + l2] - samp1_mean) / samp1_std)
                    else:
                        samples_norm2.append((self.domains[d][i * num_loops_file + loop] - samp1_mean) / samp1_std)
            # domain
            if show_domains:
                plt.clf()
                plt.figure(figsize=(8., 8.))
                ax = plt.axes(projection='3d')
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                for i in range(num_samples):
                    cm = plt.get_cmap(self.color_maps[i])
                    ax.set_prop_cycle(color=[cm(1. * (num_signals - n / 2 - 1) / num_signals) for n in range(num_signals)])
                    for j in range(num_signals):
                        zdata = [self.domains[d][i * num_loops_file + loop][j] for loop in range(num_loops_file)]
                        zdata = np.average(zdata, axis=0)
                        ax.scatter3D(self.log_freq, self.log_c[j], zdata, s=20, marker=self.simb[i],
                                     linewidths=0, alpha=3 / 5)
                for i in range(control_samples):
                    cm = plt.get_cmap(self.color_maps[num_samples + i])
                    ax.set_prop_cycle(
                        color=[cm(1. * (num_signals - n / 2 - 1) / num_signals) for n in range(num_signals)])
                    for j in range(num_signals):
                        zdata = [self.domains[d][(num_samples + i) * num_loops_file + loop][j] for loop in range(num_loops_file)]
                        zdata = np.average(zdata, axis=0)
                        ax.scatter3D(self.log_freq, self.log_c[j], zdata, s=20, marker=self.simb[num_samples + i],
                                     linewidths=0, alpha=3 / 5)
                ax.set_title(f"{self.domains_labels[d]}")
                ax.tick_params(axis="x", labelsize=9, pad=2)
                ax.tick_params(axis="y", labelsize=9, pad=2)
                ax.tick_params(axis="z", labelsize=9, pad=8)
                ax.set_xlabel('log10( frequency [Hz] )', labelpad=5)
                ax.set_ylabel('- log10( concentrations [mol/L] )', labelpad=5)
                ax.set_zlabel(f"{self.image_labels[d]}", labelpad=15)
                plt.locator_params(axis='y', nbins=num_signals)
                lgd = plt.legend(handles=self.handles, loc='upper left', ncol=1, bbox_to_anchor=(1.2, 1.04))
                plt.savefig(join(self.path, self.domains_labels[d] + ".png"), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                imgs.append(join(self.path, self.domains_labels[d] + ".png"))

            # dB domain
            if show_db:
                # check for negative z values
                all_zdata = []
                for i in range(total_samples):
                    for j in range(num_signals):
                        zdata = [self.domains[d][i * num_loops_file + loop][j] for loop in range(num_loops_file)]
                        zdata = np.average(zdata, axis=0)
                        all_zdata.extend(zdata)
                if all(n > 0 for n in all_zdata):
                    plt.clf()
                    plt.figure(figsize=(8., 8.))
                    ax_log = plt.axes(projection='3d')
                    ax_log.tick_params(axis='both', which='major', labelsize=8)
                    ax_log.tick_params(axis='both', which='minor', labelsize=8)
                    # dB domain
                    for i in range(num_samples):
                        cm = plt.get_cmap(self.color_maps[i])
                        ax_log.set_prop_cycle(color=[cm(1. * (num_signals - n / 2 - 1) / num_signals) for n in range(num_signals)])
                        for j in range(num_signals):
                            zdata = [self.domains[d][i * num_loops_file + loop][j] for loop in range(num_loops_file)]
                            zdata = np.average(zdata, axis=0)
                            ax_log.scatter3D(self.log_freq, self.log_c[j], 20 * np.log10(zdata.astype(float)), s=20,
                                             marker=self.simb[i], linewidths=0, alpha=3 / 5)
                    for i in range(control_samples):
                        cm = plt.get_cmap(self.color_maps[num_samples + i])
                        ax_log.set_prop_cycle(
                            color=[cm(1. * (num_signals - n / 2 - 1) / num_signals) for n in range(num_signals)])
                        for j in range(num_signals):
                            zdata = [self.domains[d][(num_samples + i) * num_loops_file + loop][j] for loop in range(num_loops_file)]
                            zdata = np.average(zdata, axis=0)
                            ax_log.scatter3D(self.log_freq, self.log_c[j], 20 * np.log10(zdata.astype(float)), s=20,
                                             marker=self.simb[num_samples + i], linewidths=0, alpha=3 / 5)
                    ax_log.set_title(f"{self.domains_labels[d]}")
                    ax_log.tick_params(axis="x", labelsize=9, pad=2)
                    ax_log.tick_params(axis="y", labelsize=9, pad=2)
                    ax_log.tick_params(axis="z", labelsize=9, pad=8)
                    ax_log.set_xlabel('log10( frequency [Hz] )', labelpad=5)
                    ax_log.set_ylabel('- log10( concentrations [mol/L] )', labelpad=5)
                    ax_log.set_zlabel(f"{self.domains_labels[d]} dB", labelpad=15)
                    plt.locator_params(axis='y', nbins=num_signals)
                    lgd = plt.legend(handles=self.handles, loc='upper left', ncol=1, bbox_to_anchor=(1.2, 1.04))
                    plt.savefig(join(self.path, "log " + self.domains_labels[d] + ".png"),
                                bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                    imgs.append(join(self.path, "log " + self.domains_labels[d] + ".png"))

            # domain at max variance
            if show_max_variance:
                plt.clf()
                plt.figure(figsize=(8., 8.))
                avg = np.linspace(0, 0, num_signals)
                avg1 = np.linspace(0, 0, num_signals)
                avg_signals = []  # avg samples and loops into a num_signalsXsignal_length matrix
                for j in range(num_signals):
                    same_concentration = [self.domains[d][i * num_loops_file + loop][j] for i in range(num_samples)
                                          for loop in range(num_loops_file)]
                    avg_signals.append(np.average(same_concentration, axis=0))
                var = np.var(avg_signals, axis=0)
                freq_index = np.argmax(var)
                for i in range(num_samples):
                    for loop in range(num_loops_file):
                        tmp = []
                        for j in range(num_signals):
                            tmp.append(self.domains[d][i * num_loops_file + loop][j][freq_index])
                        plt.plot(-self.c, tmp, self.simb[i], color=self.colors[i], markersize=5,
                                 label=f"sample {i + 1} loop {loop + 1}",
                                 alpha=2 / (3 + loop))
                        plt.plot(-self.x, self.__cal_curve__(tmp), color=self.colors[i], alpha=2 / (3 + loop))
                        avg = np.add(avg, tmp)
                        if loop == 0:
                            avg1 = np.add(avg1, tmp)
                for i in range(control_samples):
                    for loop in range(num_loops_file):
                        tmp = []
                        for j in range(num_signals):
                            tmp.append(self.domains[d][(num_samples + i) * num_loops_file + loop][j][freq_index])
                        plt.plot(-self.c, tmp, self.simb[num_samples + i], color=self.colors[num_samples + i],
                                 markersize=5,
                                 label=f"control {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        plt.plot(-self.x, self.__cal_curve__(tmp), color=self.colors[num_samples + i])
                for i in range(len(avg)):
                    avg[i] = avg[i] / (num_samples * num_loops_file)
                    avg1[i] = avg1[i] / num_samples
                plt.plot(-self.c, avg1, self.simb[total_samples], color=self.colors[total_samples], markersize=10,
                         label='average loop 1',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg1), color=self.colors[total_samples])
                plt.plot(-self.c, avg, self.simb[total_samples + 1], color=self.colors[total_samples + 1], markersize=8,
                         label='average',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg), color=self.colors[total_samples + 1])
                plt.title(f"{self.domains_labels[d]}({Experiment.__float_to_units__(self.freq[freq_index])}Hz)")
                plt.xlabel('log(concentration)')
                plt.ylabel(self.image_labels[d])
                lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
                title = f"{self.domains_labels[d]}_max_var.png"
                plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                imgs.append(join(self.path, title))

            # apply difrent weights and standards to data set
            # calc weights & normal avg
            avg = np.linspace(0, 0, num_signals)
            avg1 = np.linspace(0, 0, num_signals)
            weights = 0
            for i in range(num_samples):
                for loop in range(num_loops_file):
                    pca = sklearn.decomposition.PCA(n_components=2)
                    principal_components = pca.fit_transform(samples_norm1[i * num_loops_file + loop])
                    # save 1st weights
                    if i == 0:
                        if loop == 0:
                            weights = pca.components_[0]
                        else:
                            weights = weights + pca.components_[0]
                            # add principal_components to avgs
                    avg = np.add(avg, principal_components[:, 0])
                    if loop == 0:
                        avg1 = np.add(avg1, principal_components[:, 0])
                        # avg weights of 1st sample
                if i == 0:
                    for w in range(len(weights)):
                        weights[w] = weights[w] / 3
            # normal PCA1
            if show_normal_pc1:
                plt.clf()
                plt.figure(figsize=(8., 8.))
                for i in range(num_samples):
                    for loop in range(num_loops_file):
                        pca = sklearn.decomposition.PCA(n_components=2)
                        principal_components = pca.fit_transform(samples_norm1[i * num_loops_file + loop])
                        # plot points
                        plt.plot(-self.c, principal_components[:, 0], self.simb[i], color=self.colors[i], markersize=5,
                                 label=f"sample {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components[:, 0]), color=self.colors[i])
                for i in range(control_samples):
                    for loop in range(num_loops_file):
                        pca = sklearn.decomposition.PCA(n_components=2)
                        principal_components = pca.fit_transform(samples_norm1[(num_samples + i) * num_loops_file + loop])
                        # plot points
                        plt.plot(-self.c, principal_components[:, 0], self.simb[num_samples + i],
                                 color=self.colors[num_samples + i],
                                 markersize=5, label=f"control {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components[:, 0]), color=self.colors[num_samples + i])
                for i in range(len(avg)):
                    avg[i] = avg[i] / (num_samples * num_loops_file)
                    avg1[i] = avg1[i] / num_samples
                plt.plot(-self.c, avg1, self.simb[total_samples], color=self.colors[total_samples], markersize=10,
                         label='average loop 1',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg1), color=self.colors[total_samples])
                plt.plot(-self.c, avg, self.simb[total_samples + 1], color=self.colors[total_samples + 1], markersize=8,
                         label='average',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg), color=self.colors[total_samples + 1])
                plt.title('individual normalization and weights')
                plt.xlabel('log(concentration)')
                plt.ylabel('PC1(' + self.domains_labels[d] + ')')
                lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
                title = f"{self.domains_labels[d]}_individual_normalization_weights.png"
                plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                imgs.append(join(self.path, title))

            # PCA1 w 1st weights
            if show_pc1_sample1_weight:
                plt.clf()
                plt.figure(figsize=(8., 8.))
                avg = np.linspace(0, 0, num_signals)
                avg1 = np.linspace(0, 0, num_signals)
                for i in range(num_samples):
                    for loop in range(num_loops_file):
                        principal_components = []
                        for j in range(len(samples_norm1[i * num_loops_file + loop])):
                            principal_components.append(np.dot(samples_norm1[i * num_loops_file + loop][j], weights))
                            # plot
                        plt.plot(-self.c, principal_components, self.simb[i], color=self.colors[i], markersize=5,
                                 label=f"sample {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components), color=self.colors[i])
                        # add principal_components to avgs
                        avg = np.add(avg, principal_components)
                        if loop == 0:
                            avg1 = np.add(avg1, principal_components)
                for i in range(control_samples):
                    for loop in range(num_loops_file):
                        principal_components = []
                        for j in range(len(samples_norm1[(num_samples + i) * num_loops_file + loop])):
                            principal_components.append(
                                np.dot(samples_norm1[(num_samples + i) * num_loops_file + loop][j], weights))
                            # plot
                        plt.plot(-self.c, principal_components, self.simb[num_samples + i],
                                 color=self.colors[num_samples + i],
                                 markersize=5,
                                 label=f"control {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components), color=self.colors[num_samples + i])
                for i in range(len(avg)):
                    avg[i] = avg[i] / (num_samples * num_loops_file)
                    avg1[i] = avg1[i] / num_samples
                plt.plot(-self.c, avg1, self.simb[total_samples], color=self.colors[total_samples], markersize=10,
                         label='average loop 1',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg1), color=self.colors[total_samples])
                plt.plot(-self.c, avg, self.simb[total_samples + 1], color=self.colors[total_samples + 1], markersize=8,
                         label='average',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg), color=self.colors[total_samples + 1])
                plt.title('individual normalization, shared weights')
                plt.xlabel('log(concentration)')
                plt.ylabel('PC1(' + self.domains_labels[d] + ')')
                lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
                title = f"{self.domains_labels[d]}_individual_normalization_shared_weights.png"
                plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                imgs.append(join(self.path, title))

            # normal PCA1 w 1st normalization
            if show_pc1_sample1_normalization:
                plt.clf()
                plt.figure(figsize=(8., 8.))
                avg = np.linspace(0, 0, num_signals)
                avg1 = np.linspace(0, 0, num_signals)
                for i in range(num_samples):
                    for loop in range(num_loops_file):
                        pca = sklearn.decomposition.PCA(n_components=2)
                        principal_components = pca.fit_transform(samples_norm2[i * num_loops_file + loop])
                        # plot
                        plt.plot(-self.c, principal_components[:, 0], self.simb[i], color=self.colors[i], markersize=5,
                                 label=f"sample {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components[:, 0]), color=self.colors[i])
                        # add principal_components to avgs
                        avg = np.add(avg, principal_components[:, 0])
                        if loop == 0:
                            avg1 = np.add(avg1, principal_components[:, 0])
                for i in range(control_samples):
                    for loop in range(num_loops_file):
                        pca = sklearn.decomposition.PCA(n_components=2)
                        principal_components = pca.fit_transform(samples_norm2[(num_samples + i) * num_loops_file + loop])
                        # plot
                        plt.plot(-self.c, principal_components[:, 0], self.simb[num_samples + i],
                                 color=self.colors[num_samples + i],
                                 markersize=5, label=f"control {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components[:, 0]), color=self.colors[num_samples + i])
                for i in range(len(avg)):
                    avg[i] = avg[i] / (num_samples * num_loops_file)
                    avg1[i] = avg1[i] / num_samples
                plt.plot(-self.c, avg1, self.simb[total_samples], color=self.colors[total_samples], markersize=10,
                         label='average loop 1',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg1), color=self.colors[total_samples])
                plt.plot(-self.c, avg, self.simb[total_samples + 1], color=self.colors[total_samples + 1], markersize=8,
                         label='average',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg), color=self.colors[total_samples + 1])
                plt.title('shared normalization, individual weights')
                plt.xlabel('log(concentration)')
                plt.ylabel('PC1(' + self.domains_labels[d] + ')')
                lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
                title = f"{self.domains_labels[d]}_shared_normalization_individual_weights.png"
                plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                imgs.append(join(self.path, title))

            # PCA1 with 1st weight + normalization
            if show_pc1_sample1_weight_normalization:
                plt.clf()
                plt.figure(figsize=(8., 8.))
                avg = np.linspace(0, 0, num_signals)
                avg1 = np.linspace(0, 0, num_signals)
                for i in range(num_samples):
                    for loop in range(num_loops_file):
                        principal_components = []
                        for j in range(len(samples_norm2[i * num_loops_file + loop])):
                            principal_components.append(np.dot(samples_norm2[i * num_loops_file + loop][j], weights))
                            # plot
                        plt.plot(-self.c, principal_components, self.simb[i], color=self.colors[i], markersize=5,
                                 label=f"sample {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components), color=self.colors[i])
                        # add principal_components to avgs
                        avg = np.add(avg, principal_components)
                        if loop == 0:
                            avg1 = np.add(avg1, principal_components)
                for i in range(control_samples):
                    for loop in range(num_loops_file):
                        principal_components = []
                        for j in range(len(samples_norm2[(num_samples + i) * num_loops_file + loop])):
                            principal_components.append(
                                np.dot(samples_norm2[(num_samples + i) * num_loops_file + loop][j], weights))
                            # plot
                        plt.plot(-self.c, principal_components, self.simb[num_samples + i],
                                 color=self.colors[num_samples + i],
                                 markersize=5,
                                 label=f"control {i + 1} loop {loop + 1}", alpha=2 / (3 + loop))
                        # plot tend curve
                        plt.plot(-self.x, self.__cal_curve__(principal_components), color=self.colors[num_samples + i])
                for i in range(len(avg)):
                    avg[i] = avg[i] / (num_samples * num_loops_file)
                    avg1[i] = avg1[i] / num_samples
                plt.plot(-self.c, avg1, self.simb[total_samples], color=self.colors[total_samples], markersize=10,
                         label='average loop 1',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg1), color=self.colors[total_samples])
                plt.plot(-self.c, avg, self.simb[total_samples + 1], color=self.colors[total_samples + 1], markersize=8,
                         label='average',
                         alpha=3 / 4)
                plt.plot(-self.x, self.__cal_curve__(avg), color=self.colors[total_samples + 1])
                plt.title('shared normalization and weights')
                plt.xlabel('log(concentration)')
                plt.ylabel('PC1(' + self.domains_labels[d] + ')')
                lgd = plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
                title = f"{self.domains_labels[d]}_shared_normalization_weights.png"
                plt.savefig(join(self.path, title), bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)
                imgs.append(join(self.path, title))
        return imgs

    def plt_normalized_m0(self):
        # create local vars
        num_loops_file = self.num_loops_file
        num_samples = self.num_samples
        num_signals = self.num_signals
        control_samples = self.total_samples - num_samples
        for d in range(self.num_domains):
            # specter /M0
            for i in range(num_samples):
                for j in range(num_signals - 1):
                    for loop in range(num_loops_file):
                        plt.plot(self.freq,
                                 [a / b for a, b in zip(self.domains[d][i * num_loops_file + loop][j],
                                                        self.domains[d][i * num_loops_file + loop][
                                                            num_signals - 1])],
                                 self.simb[i], color=self.colors[j], markersize=5,
                                 label=f"{self.signals[j]}_s{i + 1}_l{loop + 1}",
                                 alpha=2 / (3 + loop))
            for i in range(control_samples):
                for j in range(num_signals - 1):
                    for loop in range(num_loops_file):
                        plt.plot(self.freq, [a / b for a, b in
                                             zip(self.domains[d][(num_samples + i) * num_loops_file + loop][j],
                                                 self.domains[d][(num_samples + i) * num_loops_file + loop][
                                                     num_signals - 1])], self.simb[num_samples + i],
                                 color=self.colors[j], markersize=5, label=f"{self.signals[j]}_control_l{loop}",
                                 alpha=2 / (3 + loop))
            plt.title(f"{self.domains_labels[d]} / M0")
            plt.xlabel('frequency [Hz]')
            plt.xscale('log')
            plt.ylabel(self.image_labels[d])
            plt.legend(loc='upper left', ncol=int(num_signals / 2), bbox_to_anchor=(1.04, 1.04))
            # plt.show()
            # specter -M0
            for i in range(num_samples):
                for j in range(num_signals - 1):
                    for loop in range(num_loops_file):
                        plt.plot(self.freq,
                                 [a - b for a, b in zip(self.domains[d][i * num_loops_file + loop][j],
                                                        self.domains[d][i * num_loops_file + loop][
                                                            num_signals - 1])],
                                 self.simb[i], color=self.colors[j], markersize=5,
                                 label=f"{self.signals[j]}_s{i + 1}_l{loop + 1}",
                                 alpha=2 / (3 + loop))
            for i in range(control_samples):
                for j in range(num_signals - 1):
                    for loop in range(num_loops_file):
                        plt.plot(self.freq, [a - b for a, b in
                                             zip(self.domains[d][(num_samples + i) * num_loops_file + loop][j],
                                                 self.domains[d][(num_samples + i) * num_loops_file + loop][
                                                     num_signals - 1])], self.simb[num_samples + i],
                                 color=self.colors[j], markersize=5, label=f"{self.signals[j]}_control_l{loop}",
                                 alpha=2 / (3 + loop))
            plt.title(f"{self.domains_labels[d]} - M0")
            plt.xlabel('frequency [Hz]')
            plt.xscale('log')
            plt.ylabel(self.image_labels[d])
            plt.legend(loc='upper left', ncol=int(num_signals / 2), bbox_to_anchor=(1.04, 1.04))
            # plt.show()

    def show_nyquist(self):
        # creat local vars
        num_loops_file = self.num_loops_file
        num_samples = self.num_samples
        # apply difrent weights and standards to data set
        # n normalization
        color = ['tab:blue', 'tab:orange', 'tab:green']
        for i in range(num_samples):
            for loop in range(num_loops_file):
                for j in range(len(self.isamples[0])):
                    plt.plot(self.rsamples[i * num_loops_file + loop][j][1:8],
                             self.isamples[i * num_loops_file + loop][j][1:8], self.simb[i],
                             markersize=10 / (i + 1), color=color[i])
                    plt.plot(self.rsamples[i * num_loops_file + loop][j][1:8],
                             self.isamples[i * num_loops_file + loop][j][1:8],
                             color=color[i])
        plt.title("Nyquist")
        plt.ylabel("I{Z} [ohm]")
        plt.xlabel("R{Z} [ohm]")
        # plt.show()
        nsamples = []
        for i in range(num_samples):
            index = []
            for j in range(len(self.isamples[0])):
                # plt
                plt.plot(self.rsamples[i * num_loops_file][j][1:8], self.isamples[i * num_loops_file][j][1:8],
                         label=self.signals[j])
                plt.plot(self.rsamples[i * num_loops_file][j][1:8], self.isamples[i * num_loops_file][j][1:8],
                         self.simb[i],
                         markersize=10 / (i + 1), color=color[i])
                # get max
                index.append(self.rsamples[i * num_loops_file][j][
                                 self.isamples[i * num_loops_file][j][1:8].index(
                                     max(self.isamples[i * num_loops_file][j][1:8]))])
                # plt Nyquist
            plt.title("Nyquist sample" + str(i + 1))
            plt.ylabel("I{Z} [ohm]")
            plt.xlabel("R{Z} [ohm]")
            plt.legend(bbox_to_anchor=(1, 1.05))
            # plt.show()
            # plt max
            plt.plot(-self.c, index, color=color[i])
            plt.plot(-self.c, index, self.simb[i], markersize=10 / (i + 1), color=color[i])
            plt.title("Nyquist sample" + str(i + 1))
            plt.ylabel("R{Z} [ohm]")
            plt.xlabel("log(concentration)")
            # plt.show()
            nsamples.append(index)
            # plt all max
        for i, niq in enumerate(nsamples):
            plt.plot(-self.c, niq, color=color[i])
            plt.plot(-self.c, niq, self.simb[i], markersize=10 / (i + 1), color=color[i])
        plt.title("Nyquist all sample")
        plt.ylabel("R{Z} [ohm]")
        plt.xlabel("log(concentration)")
        # plt.show()

    def show_samples_less_control(self):  # loss tg 100Hz-1Hz
        # creat local vars
        num_loops_file = self.num_loops_file
        num_samples = self.num_samples
        d = len(self.domains_labels) - 1  # point to tg
        # calc & plt
        avg = []
        for i in range(num_samples):
            for loop in range(num_loops_file):
                tmp = []
                for j in range(len(self.tsamples[i * num_loops_file + loop])):
                    tmp.append(
                        self.tsamples[i * num_loops_file + loop][j][self.freq.tolist().index(1)] -
                        self.tsamples[i * num_loops_file + loop][j][
                            self.freq.tolist().index(100)])
                plt.plot(-self.c, tmp, self.simb[i], color=self.colors[i], markersize=5,
                         label='sample ' + str(i + 1) + '_l' + str(loop + 1),
                         alpha=2 / (3 + loop))
                if i == 0 and loop == 0:
                    avg = tmp
                else:
                    avg = np.add(avg, tmp)
        for i in range(len(avg)):
            avg[i] = avg[i] / (num_samples * num_loops_file)
        plt.plot(-self.c, avg, self.simb[num_samples], color=self.colors[num_samples], markersize=10, label='avg')
        plt.plot(-self.x, self.__cal_curve__(avg), color=self.colors[num_samples])
        plt.title('loss tg 100Hz-1Hz')
        plt.xlabel('log(concentration)')
        plt.ylabel('PC1(' + self.domains_labels[d] + ')')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
        # plt.show()
