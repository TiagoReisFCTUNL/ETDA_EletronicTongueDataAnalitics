# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:05:01 2022

@author: Asus
"""

# TODO test other than PCA

# IMPORT
import os
#import sklearn.decomposition
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from math import cos
from math import sin
from math import tan
from math import pi
import re


class Project:
    # DEF
    @staticmethod
    def __convert(text):
        return int(text) if text.isdigit() else text.lower()

    @staticmethod
    def __alphanum_key(key):
        return [Project.__convert(c) for c in re.split('([0-9]+)', key)]

    @staticmethod
    def sorted_alphanumeric(data):
        return sorted(data, key=Project.__alphanum_key)

    @staticmethod
    def line_func(x, a, b):
        return a * x + b

    @staticmethod
    def lin_surface_func(data, a, b, c):
        return data[0] * a + data[1] * b + c

    @staticmethod
    def invert(array):
        return array[::-1]

    def cal_curve(self, pc1):
        concentrations = self.c.copy()[:-1] if self.calibrate_with_control_concentration == 0 else self.c.copy()
        data1 = pc1.copy()[:-1] if self.calibrate_with_control_concentration == 0 else pc1.copy()
        popt, pcov, *_ = curve_fit(Project.line_func, concentrations, data1)
        return popt[0] * self.z + popt[1]

    def cal_surface(self, pc1, pc2, resolution, c):
        x = np.linspace(np.min(pc1), np.max(pc1), resolution)
        y = np.linspace(np.min(pc2), np.max(pc2), resolution)
        xs, ys = np.meshgrid(x, y, sparse=True)
        concentrations = c.copy()[:-1] if not self.calibrate_with_control_concentration else c.copy()
        data1 = np.array(pc1.copy()).reshape(
            (self.experiment_samples * self.num_loops_file, self.num_concentrations))
        data2 = np.array(pc2.copy()).reshape(
            (self.experiment_samples * self.num_loops_file, self.num_concentrations))
        data1 = data1[:, :len(concentrations)]
        data2 = data2[:, :len(concentrations)]
        data1 = data1.reshape(
            (1, len(pc1) - self.experiment_samples * self.num_loops_file * (1 - self.calibrate_with_control_concentration)))[0]
        data2 = data2.reshape(
            (1, len(pc2) - self.experiment_samples * self.num_loops_file * (1 - self.calibrate_with_control_concentration)))[0]
        concentrations = - np.resize(concentrations,
                                     (1, len(concentrations) * self.experiment_samples * self.num_loops_file))[0]
        popt, pcov, *_ = curve_fit(Project.lin_surface_func, [data1, data2], concentrations)
        self.surface_sensibility1 = popt[0]
        self.surface_sensibility2 = popt[1]
        self.surface_constant = popt[2]
        return Project.lin_surface_func((xs, ys), popt[0], popt[1], popt[2])

    def __init__(self, project_name, domains_labels, project_samples,
                 experiment_list, folder, experiments, calibrate_with_control_concentration=True):
        # VALS
        self.project_name = project_name
        self.domains_labels = domains_labels  # ['losses_tangent']
        self.calibrate_with_control_concentration = calibrate_with_control_concentration  # 0
        self.folder = folder

        # define experiments
        self.experiments = experiments
        self.num_environments = len(experiments)
        self.num_domains = len(domains_labels)
        # define domain bools
        impedance = 'impedance' in domains_labels
        phase = 'phase' in domains_labels
        real_impedance = 'real_impedance' in domains_labels
        imaginary_impedance = 'imaginary_impedance' in domains_labels
        losses_tangent = 'losses_tangent' in domains_labels

        # define environment_signals
        self.environment_signals = []  # environment_signals[experiment][samples][concentrations]
        self.environments_folders = []  # environments_folders[experiment][samples]
        for f in experiments:
            v = []
            folders = []
            tmp = os.listdir(join(folder, f))
            # identify folders
            for f2 in tmp:
                if str(f2).find('.') == -1:
                    folders.append(f2)
                    v2 = []
                    # save environment_signals for each folder
                    tmp2 = os.listdir(join(str(folder), str(f), str(f2)))
                    for file in tmp2:
                        if file.find('.csv') != -1:
                            tmp3 = file
                            # filter temp3
                            if tmp3.find('_') != -1:
                                tmp3 = tmp3[:tmp3.find('_')]
                            else:
                                tmp3 = tmp3[:tmp3.find('.')]
                            v2.append(tmp3)
                    v2 = list(dict.fromkeys(v2))  # remove duplicates (normal + corrected)
                    v2 = Project.sorted_alphanumeric(v2)
                    tmp4 = v2[1:]
                    tmp4.append(v2[0])
                    v2 = tmp4
                    v.append(v2)
            self.environment_signals.append(v)
            # sort control samples first
            control_samples = Project.invert(project_samples[f][0])  # invert preserves order
            for sample in control_samples:
                folders.remove(sample)
                folders.insert(0, sample)
            self.environments_folders.append(folders)

        # get index for smallest set of experiment samples
        num_control_samples = [len(all_samples[0]) for experiment, all_samples in project_samples.items()]
        index = 0
        min_experiment_samples = len(self.environment_signals[0]) - num_control_samples[0]
        for i in range(1, self.num_environments):
            curr_experiment_samples = len(self.environment_signals[i]) - num_control_samples[i]
            if min_experiment_samples > curr_experiment_samples:
                index = i
                min_experiment_samples = curr_experiment_samples
        # set 1st concentration and skip
        self.experiment_samples = min_experiment_samples  # 3
        self.first_concentration = experiment_list[index][0]  # 9
        self.skip = experiment_list[index][1]  # 3
        self.control_samples = num_control_samples  # int list
        # check for smallest concentrations
        self.num_concentrations = len(self.environment_signals[0][0])
        v_tmp = self.environment_signals[0][0]
        for i in range(self.num_environments):
            for signals in self.environment_signals[i]:
                if self.num_concentrations > len(signals):
                    self.num_concentrations = len(signals)
                    v_tmp = signals
        # environment_signals = [smallest set of concentrations]
        self.environment_signals = v_tmp

        # read .csv files
        self.freq = []  # freq[num_environments][signal_length]
        self.environments = []
        self.signal_length = 0
        self.total_length = 0
        self.num_loops_file = 0
        for e in range(self.num_environments):
            # read files water1
            num_samples = self.experiment_samples + self.control_samples[e]
            samples = []
            samples_phase = []
            csv_to_open = ""
            curr_signal_length = 0
            for i in range(num_samples):
                signals_out = []
                signals_out_phase = []
                for s in self.environment_signals:
                    csv_to_open = join(folder, experiments[e], self.environments_folders[e][i], s + '_corrected.csv')
                    csv = (pd.read_csv(csv_to_open, skiprows=2, sep=";")).to_numpy()
                    curr_signal_length = csv[:, 1].tolist().count(1)
                    signals_out.append(csv[:, 10])
                    signals_out_phase.append(csv[:, 11])
                samples.append(signals_out)
                samples_phase.append(signals_out_phase)

            freq_e = (pd.read_csv(csv_to_open, skiprows=2, sep=";")).to_numpy()[:curr_signal_length, 4]
            self.freq.append(freq_e.copy())
            curr_num_loops_file = int(samples[0][0].size / freq_e.size)
            if e == 0:
                # set const
                self.total_length = samples[0][0].size
                self.num_loops_file = int((pd.read_csv(csv_to_open, skiprows=2, sep=";")).to_numpy()[:, 4]
                                          .size / curr_signal_length)
            elif curr_num_loops_file < self.num_loops_file:
                if curr_num_loops_file:
                    self.num_loops_file = curr_num_loops_file
                else:
                    raise Exception(f"Bad file/s in: {experiments[e]}")

            # convert degrees to rads
            for i in range(num_samples):
                for j in range(self.num_concentrations):
                    for f in range(curr_signal_length * self.num_loops_file):
                        samples_phase[i][j][f] = samples_phase[i][j][f] * (pi / 180)
            # get I{Z} and R{Z}
            imaginary = []
            real = []
            tangent = []
            for i in range(num_samples):
                imaginary_sample = []
                real_sample = []
                tangent_sample = []
                for j in range(self.num_concentrations):
                    temp_i = []
                    temp_r = []
                    temp_t = []
                    for f in range(curr_signal_length * self.num_loops_file):
                        temp_i.append(samples[i][j][f] * sin(samples_phase[i][j][f]))
                        temp_r.append(samples[i][j][f] * cos(samples_phase[i][j][f]))
                        temp_t.append(tan(samples_phase[i][j][f]))
                    imaginary_sample.append(temp_i)
                    real_sample.append(temp_r)
                    tangent_sample.append(temp_t)
                imaginary.append(imaginary_sample)
                real.append(real_sample)
                tangent.append(tangent_sample)
            # save selected domains
            tmp = []
            if impedance:
                tmp.append(samples)
            if phase:
                tmp.append(samples_phase)
            if real_impedance:
                tmp.append(real)
            if imaginary_impedance:
                tmp.append(imaginary)
            if losses_tangent:
                tmp.append(tangent)
            self.environments.append(tmp)

        if len(self.environments) != 0:
            self.num_characteristics = len(self.environments[0])
        else:
            self.num_characteristics = 0

        # ready data_set
        data_set = []
        # line_length = num_environments * num_characteristics * signal_length  #not including control
        for i in range(self.experiment_samples):
            for loop in range(self.num_loops_file):
                for j in range(self.num_concentrations):
                    line = []
                    for e1 in range(self.num_environments):
                        curr_sample = i + self.control_samples[e1]
                        for e2 in range(self.num_characteristics):  # characteristics (Z, o, R{Z}, I{Z}, ...)
                            # for each experiment, all samples share the same control
                            signal_length = len(self.freq[e1])
                            for f in range(signal_length):
                                line.append(self.environments[e1][e2][curr_sample][j][loop * signal_length + f])
                                # add control to line
                                for i2 in range(self.control_samples[e1]):
                                    line.append(self.environments[e1][e2][i2][j][loop * signal_length + f])
                    data_set.append(line)
            # normalize data set
        self.scaledData = StandardScaler().fit_transform(data_set)

        # calc PCA
        # condense the 2 environment
        self.pca = PCA(n_components=2)
        principal_components = self.pca.fit_transform(self.scaledData)
        principal_components[:, 0] = -principal_components[:, 0]
        self.principal_components = principal_components
        c = np.arange(start=self.first_concentration, stop=self.first_concentration + (self.num_concentrations - 1) * self.skip,
                      step=self.skip)
        self.c = np.concatenate((c, [18]))
        self.x = np.linspace(np.min(principal_components[:, 0]) - 1, np.max(principal_components[:, 0]) + 1, 101)
        self.y = np.linspace(np.min(principal_components[:, 1]) - 1, np.max(principal_components[:, 1]) + 1, 101)
        self.z = np.arange(start=self.first_concentration, stop=self.first_concentration + (self.num_concentrations - 1) * self.skip,
                           step=0.1)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.surface_sensibility1 = 0
        self.surface_sensibility2 = 0
        self.surface_constant = 0
        self.zs = Project.cal_surface(self, principal_components[:, 0], principal_components[:, 1], 101, self.c)

    def export_pca(self):
        # ready columns
        columns = "Sample;"
        for i, e in enumerate(self.experiments):
            for d in self.domains_labels:
                signal_length = len(self.freq[i])
                for f in range(signal_length):
                    columns += e + ', ' + d + "(" + str(self.freq[i][f]) + ");"
                    columns += 'Control[' + e + ', ' + d + "(" + str(self.freq[i][f]) + ")];"
        columns += '\n'
        # save lines
        with open(join(self.folder, 'PCA.csv'), "w") as PCA_file:
            PCA_file.writelines(columns)
            line_counter = 0
            for i in range(self.experiment_samples):
                for loop in range(self.num_loops_file):
                    for j in range(self.num_concentrations):
                        e1_w = 0
                        for e1 in range(self.num_environments):
                            line = self.environment_signals[j] + ';'
                            signal_length = len(self.freq[e1])
                            for e2 in range(self.num_characteristics):  # characteristics (Z, o, R{Z}, I{Z}, ...)
                                # set vars
                                num_columns = 1 + self.control_samples[e1]
                                start_index = e1_w + e2 * signal_length * num_columns
                                for f in range(signal_length):
                                    line += str(
                                        self.scaledData[line_counter][start_index + f * num_columns]) + ';'
                                    # add control to line
                                    for i2 in range(self.control_samples[e1]):
                                        line += str(
                                            self.scaledData[line_counter][start_index + f * num_columns + 1 + i2]) + ';'
                            e1_w += self.num_characteristics * signal_length
                        line += '\n'
                        line_counter += 1
                        PCA_file.writelines(line)

    def create_heatmap(self):
        # setup colors
        cm = plt.get_cmap('gist_rainbow')
        color = [cm(1. * n / self.num_concentrations) for n in range(self.num_concentrations)]
        # setup handles
        handles = [
            mlines.Line2D([], [], color=color[i], marker='.',
                          linestyle='None', markersize=5,
                          label=f"{self.environment_signals[i]}") for i in range(self.num_concentrations)
        ]
        # create figure
        fig = plt.figure(figsize=(8., 8.))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        ax.set_prop_cycle(color=color)
        plt.contourf(self.x, self.y, self.zs, levels=200, cmap=plt.cm.Greys)
        ax.grid()
        # plot points
        count = 0
        for i in range(self.experiment_samples):
            for loop in range(self.num_loops_file):
                for j in range(self.num_concentrations):
                    ax.scatter(self.principal_components[:, 0][count], self.principal_components[:, 1][count], s=50)
                    count += 1
        # plot avg points
        points_concentration = self.experiment_samples * self.num_loops_file
        for j in range(self.num_concentrations):
            avg_x = sum(self.principal_components[:, 0][i * self.num_concentrations + j] for i in range(points_concentration)) / points_concentration
            avg_y = sum(self.principal_components[:, 1][i * self.num_concentrations + j] for i in range(points_concentration)) / points_concentration
            ax.scatter(avg_x, avg_y, marker='x', s=100)
        # finish figure
        ax.axis('scaled')
        plt.colorbar(orientation="horizontal")
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.04, 1.04))
        plt.text(1, 0.2,
                 f" PC1 sensibility: {round(self.surface_sensibility1, 3)}\n"
                 f" PC2 sensibility: {round(self.surface_sensibility2, 3)}\n"
                 f" surface_constant: {round(self.surface_constant, 3)}",
                 horizontalalignment='left',
                 transform=ax.transAxes)
        # save figure
        plt.savefig(join(self.folder, "pca.png"), bbox_inches='tight', dpi=100)

    def show_weights(self):  # plt PCA calculations
        simb = ['.', '+', '_']
        cm = plt.get_cmap('gist_rainbow')
        color = [cm(1. * i / self.num_concentrations) for i in range(self.num_concentrations)]
        plt.rc('figure', max_open_warning=0)
        # scale weights vector by its max
        weights1 = self.pca.components_[0]
        weights1 = weights1 / abs(max(weights1, key=abs))
        weights2 = self.pca.components_[1]
        weights2 = weights2 / abs(max(weights2, key=abs))
        weights = [weights1, weights2]
        e_w = 0
        imgs = []
        for e in range(self.num_environments):
            signal_length = len(self.freq[e])
            for e2 in range(self.num_characteristics):
                # set vars
                num_columns = 1 + self.control_samples[e]  # number of samples in columns per experiment and domain
                start_index = e_w + e2 * signal_length * num_columns
                # film column
                fig, ax1 = plt.subplots(figsize=(8., 6.))
                ax2 = ax1.twinx()
                count = 0
                for i in range(self.experiment_samples):
                    for loop in range(self.num_loops_file):
                        for j in range(self.num_concentrations):
                            line = self.scaledData[count]
                            ax1.plot(self.freq[e],
                                     [line[start_index + n * num_columns] for n in range(signal_length)],
                                     simb[loop], color=color[j], markersize=3, alpha=.5)
                            count += 1
                ax2.plot(self.freq[e], [weights[0][start_index + w * num_columns] for w in range(signal_length)], color='b')
                ax2.plot(self.freq[e], [weights[1][start_index + w * num_columns] for w in range(signal_length)], color='r')
                title = f"PCA calculations (film {self.experiments[e]}) ({self.domains_labels[e2]})"
                plt.title(title)
                plt.xscale('log')
                ax1.set_xlabel('frequency [Hz]')
                ax1.set_ylabel('data')
                ax2.set_ylabel('weights')
                ax2.set_ylim([-1.1, 1.1])
                plt.savefig(join(self.folder, f"{title}.png"), bbox_inches='tight', dpi=100)
                imgs.append(join(self.folder, f"{title}.png"))
                # control columns
                for control in range(self.control_samples[e]):
                    fig, ax1 = plt.subplots(figsize=(8., 6.))
                    ax2 = ax1.twinx()
                    count = 0
                    for i in range(self.experiment_samples):
                        for loop in range(self.num_loops_file):
                            for j in range(self.num_concentrations):
                                line = self.scaledData[count]
                                ax1.plot(self.freq[e],
                                         [line[start_index + n * num_columns + 1 + control] for n in range(signal_length)],
                                         simb[loop], color=color[j], markersize=3, alpha=.5)
                                count += 1
                    ax2.plot(self.freq[e], [weights[0][start_index + w * num_columns + 1 + control] for w in range(signal_length)], color='b')
                    ax2.plot(self.freq[e], [weights[1][start_index + w * num_columns + 1 + control] for w in range(signal_length)], color='r')
                    title = f"PCA calculations (control {self.environments_folders[e][control]}) ({self.domains_labels[e2]})"
                    plt.title(title)
                    plt.xscale('log')
                    ax1.set_xlabel('frequency [Hz]')
                    ax1.set_ylabel('data')
                    ax2.set_ylabel('weights')
                    ax2.set_ylim([-1.1, 1.1])
                    plt.savefig(join(self.folder, f"{title}.png"), bbox_inches='tight', dpi=100)
                    imgs.append(join(self.folder, f"{title}.png"))
            e_w += self.num_characteristics * signal_length
        return imgs

    def create_3d_view(self, plt_3d_trend_line):  # plt PC1(log(C))
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        cm = plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle(
            color=[cm(1. * n / (self.experiment_samples * self.num_loops_file)) for n in
                   range(self.experiment_samples * self.num_loops_file)])
        # PC1
        pc1_avg = np.linspace(0, 0, self.num_concentrations)
        pc1_avg1 = np.linspace(0, 0, self.num_concentrations)
        pc2_avg = np.linspace(0, 0, self.num_concentrations)
        pc2_avg1 = np.linspace(0, 0, self.num_concentrations)
        for i in range(self.experiment_samples):
            for loop in range(self.num_loops_file):
                index = (i * self.num_loops_file + loop) * self.num_concentrations
                # plot
                ax.scatter3D(self.principal_components[:, 1][index:index + self.num_concentrations],
                             self.principal_components[:, 0][index:index + self.num_concentrations], -self.c, s=20,
                             linewidths=0,
                             label='sample ' + str(i) + ' loop ' + str(loop + 1))  # , alpha=2/(3+l)
                # add principal_components to avgs
                pc1_avg = np.add(pc1_avg, self.principal_components[:, 0][index:index + self.num_concentrations])
                pc2_avg = np.add(pc2_avg, self.principal_components[:, 1][index:index + self.num_concentrations])
                if loop == 0:
                    pc1_avg1 = np.add(pc1_avg1, self.principal_components[:, 0][index:index + self.num_concentrations])
                    pc2_avg1 = np.add(pc2_avg1, self.principal_components[:, 1][index:index + self.num_concentrations])
        if plt_3d_trend_line:
            for i in range(len(pc1_avg)):
                pc1_avg[i] = pc1_avg[i] / (self.experiment_samples * self.num_loops_file)
                pc1_avg1[i] = pc1_avg1[i] / self.experiment_samples
                pc2_avg[i] = pc2_avg[i] / (self.experiment_samples * self.num_loops_file)
                pc2_avg1[i] = pc2_avg1[i] / self.experiment_samples
            cm = plt.get_cmap('Greys')
            color = [cm(1. * (i + 1) / 3) for i in range(2)]
            ax.set_prop_cycle(color=color)
            ax.scatter3D(pc1_avg, pc2_avg, -self.c, s=50, marker='s', label='average', alpha=3 / 4)
            ax.plot(Project.cal_curve(self, pc1_avg), Project.cal_curve(self, pc2_avg), -self.z, color=color[0])
            ax.scatter3D(pc1_avg1, pc2_avg1, -self.c, s=70, marker='*', label='average loop 1', alpha=3 / 4)
            ax.plot(Project.cal_curve(self, pc1_avg1), Project.cal_curve(self, pc2_avg1), -self.z, color=color[1])

        ax.plot_surface(self.yy, self.xx, self.zs, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('PCA(log(C))')
        ax.set_xlabel('PC2')
        ax.set_ylabel('PC1')
        ax.set_zlabel('- log10( concentrations [mol/L] )')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.04))
        return join(self.folder, "PCA(log(C)).png")
