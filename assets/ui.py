
import sys
from assets.project import Project
from assets.experiment import Experiment
from assets.sample import Sample
from tkinter import ttk
from tkinter import *
from PIL import ImageTk, Image
from os.path import exists, join
from os import mkdir, getcwd, walk
from shutil import copytree, rmtree, copy
from tkinter import filedialog
from tkinter.messagebox import showinfo, showwarning, showerror, askquestion
from tkinter.simpledialog import askstring
import csv


class ScrolledFrame:
    def on_resize(self, event):
        bbox = self.canvas.bbox(ALL)
        self.canvas.config(width=bbox[2], scrollregion=bbox)

    def resize_frame(self, event):
        self.canvas.itemconfig(self.frame_window, width=event.width)

    def on_mouse_wheel(self, event):
        # better checking whether event happens inside frame
        self.canvas.yview_scroll(event.delta // -30, 'units')

    def reset_scroll(self):
        self.canvas.yview_moveto(0.0)

    def get_frame(self):
        return self.frame

    def __init__(self, parent):
        # note: Canvas is the outer container
        self.canvas = Canvas(parent, highlightthickness=0)
        # *** modify the below line to suit your layout manager
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        sb = Scrollbar(parent, orient=VERTICAL, command=self.canvas.yview)
        # *** modify the below line to use same layout manager as canvas
        sb.pack(side=RIGHT, fill=Y)

        self.canvas.config(yscrollcommand=sb.set)

        self.frame = Frame(self.canvas)
        self.frame.columnconfigure(0, weight=1)
        self.frame_window = self.canvas.create_window(0, 0, window=self.frame, anchor='nw')
        self.canvas.bind("<Configure>", self.resize_frame)
        self.frame.bind('<Configure>', self.on_resize)
        # use bind_all() to make sure mouse wheel events can be triggered
        # even the canvas is filled with labels on top
        self.canvas.bind_all('<MouseWheel>', self.on_mouse_wheel)


class UI:
    def __export_images__(self):
        if not self.last_images:
            showerror('Error', 'No images found.\nPlease create the using the run button.')
            return
        try:
            for img in self.last_images:
                copy(img, self.experiment_dir.get())
            showinfo('Info', f"Image/s successfully exported to:\n{self.experiment_dir.get()}")
        except Exception as error:
            showerror('Error', str(error))

    def __load_experiment_frame__(self, frame_settings):  # frame_settings = [[name, value, identifier], ...]
        # get num items
        last_item = len(frame_settings) - 1
        # set experiment_name
        experiment_name = frame_settings[0][0]
        # create experiment frame
        tmp_frame = Frame(self.project_frame)
        tmp_frame.pack(fill=X, padx=10, pady=5)
        tmp_frame.grid_columnconfigure(1, weight=1)
        # add frame to frame dict
        self.project_frames[experiment_name] = tmp_frame
        for index, settings in enumerate(frame_settings):
            # fill frame
            if index == 0:  # experiment
                ttk.Radiobutton(tmp_frame, text=settings[0], variable=self.project_item, value=settings[1],
                                style='IndicatorOff.TRadiobutton').grid(row=0, column=0, columnspan=2, pady=(0, 3),
                                                                        ipadx=10, sticky=E + W)
            elif index == last_item:  # last sample
                Label(tmp_frame, width=10, padx=0, pady=0, image=self.img_u251x4).grid(row=index, column=0, padx=2,
                                                                                       sticky=N + W + S + E)
                ttk.Radiobutton(tmp_frame, text=settings[0], variable=self.project_item, value=settings[1],
                                style='IndicatorOff.TRadiobutton').grid(row=index, column=1, pady=3, sticky=W + E)
            else:  # sample
                Label(tmp_frame, width=10, padx=0, pady=0, image=self.img_u252x0).grid(row=index, column=0, padx=2,
                                                                                       sticky=N + W + S + E)
                ttk.Radiobutton(tmp_frame, text=settings[0], variable=self.project_item, value=settings[1],
                                style='IndicatorOff.TRadiobutton').grid(row=index, column=1, pady=3, sticky=W + E)

    def __create_experiment_frame__(self, experiment_name, frame_num, samples):
        # create experiment frame
        tmp_frame = Frame(self.project_frame)
        tmp_frame.pack(fill=X, padx=10, pady=5)
        tmp_frame.grid_columnconfigure(1, weight=1)
        # add frame to frame dict
        self.project_frames[experiment_name] = tmp_frame
        # add experiment rb
        ttk.Radiobutton(tmp_frame, text=experiment_name, variable=self.project_item,
                        value=f"{{'type': 'experiment', 'value': '{experiment_name}'}}",
                        style='IndicatorOff.TRadiobutton').grid(row=0, column=0, columnspan=2, pady=(0, 3), ipadx=10,
                                                                sticky=E + W)
        self.project_list.append([experiment_name, f"{{'type': 'experiment', 'value': '{experiment_name}'}}", frame_num])
        # add samples
        num_samples = len(samples)
        for i, sample in enumerate(samples):
            # add index lbl
            if i < num_samples - 1:
                Label(tmp_frame, width=10, padx=0, pady=0, image=self.img_u252x0).grid(row=i + 1, column=0, padx=2,
                                                                                       sticky=N + W + S + E)
            else:
                Label(tmp_frame, width=10, padx=0, pady=0, image=self.img_u251x4).grid(row=i + 1, column=0, padx=2,
                                                                                       sticky=N + W + S + E)
            # add rb
            ttk.Radiobutton(tmp_frame, text=sample, variable=self.project_item,
                            value=f"{{'type': 'sample', 'value': ('{experiment_name}', '{sample}')}}",
                            style='IndicatorOff.TRadiobutton').grid(row=1 + i, column=1, pady=3, sticky=W + E)
            self.project_list.append(
                [sample, f"{{'type': 'sample', 'value': ('{experiment_name}', '{sample}')}}", frame_num])

    def __save_project_list__(self, trg_path):  # str trg_path (trg_path points to .csv)
        with open(trg_path, "w", newline='') as the_file:
            writer = csv.writer(the_file)
            for p_list in self.project_list:
                writer.writerow(p_list)

    def __file_execute__(self, clicked):
        self.file_clicked.set("File")
        # new project
        if clicked == "new project":
            # ask new project name
            filename = askstring('New project', 'name:\t\t\t\t\t')
            if filename:
                # check if trg_path exists
                trg_path = join(self.current_folder, filename)
                if exists(trg_path):
                    # check if current==new
                    if self.project_dir == filename:
                        showinfo('Invalid name', 'Already in project.')
                    # show dup msg
                    elif askquestion('Invalid name', f"Project already exists.\nLoad project {filename}?") == 'yes':
                        # set trg_path to project csv
                        trg_path = join(trg_path, filename + ".csv")
                        # check if .csv exists
                        if not exists(trg_path):
                            showerror('Error', 'No save file from selected project was found.')
                        else:
                            # clear project_list
                            self.project_list = []
                            # clear self.project_frames
                            self.project_frames = {}
                            # select current project as new project
                            self.project_dir = filename
                            # load settings
                            self.__load_settings__()
                            # clear current view
                            for widget in self.project_frame.winfo_children():
                                widget.destroy()
                            for widget in self.data_frame.winfo_children():
                                widget.destroy()
                            # update project_lbl
                            ttk.Radiobutton(self.project_frame, text=f" current project: {filename} ", variable=self.project_item,
                                            value=f"{{'type': 'project', 'value': '{filename}'}}",
                                            style='IndicatorOff.TRadiobutton').pack(pady=10)
                            # get project_list
                            with open(trg_path, "r", newline='') as the_file:
                                reader = csv.reader(the_file)
                                for p_list in reader:
                                    self.project_list.append(p_list)
                            # get numbers of the experiments
                            num_experiments = list(
                                set([int(p_list[2]) for p_list in self.project_list]))  # list with unique identifiers
                            # def immediate experiments
                            immediate_samples = next(walk(join(self.current_folder, filename)))[1]
                            # def saved experiments
                            saved_experiments = [p_list[0] for p_list in self.project_list if
                                                 eval(p_list[1])['type'] == 'experiment']
                            # check if project_list contains all immediate dirs
                            if not all([item in saved_experiments for item in immediate_samples]):
                                # identify missing experiments
                                m_experiments = [item for item in immediate_samples if item not in saved_experiments]
                                # resolve answer
                                if askquestion('Warning',
                                               "There were found experiment(s) not saved.\nWant to recover them?") == 'yes':
                                    for experiment_name in m_experiments:
                                        # fill frame_settings from dir
                                        # get current identifier for project_list
                                        frame_num = str(max(num_experiments) + 1)  # frame numbers start at 0
                                        # get samples
                                        samples = next(walk(join(self.current_folder, filename, experiment_name)))[1]
                                        # compound
                                        frame_settings = [[experiment_name,
                                                           f"{{'type': 'experiment', 'value': '{experiment_name}'}}",
                                                           frame_num]]
                                        for sample in samples:
                                            frame_settings.append([sample,
                                                                   f"{{'type': 'sample', 'value': ('{experiment_name}', '{sample}')}}",
                                                                   frame_num])
                                        # add to project_list
                                        self.project_list.extend(frame_settings)
                                        # update num_experiments
                                        num_experiments.append(int(frame_num))
                                        # update csv
                                        self.__save_project_list__(trg_path)
                                else:
                                    # delete unsaved folders
                                    for experiment_name in m_experiments:
                                        rmtree(join(self.current_folder, filename, experiment_name))
                            # check if all immediate dirs in project_list exists
                            # def m_experiments
                            m_experiments = []
                            m_indexes = []
                            for i in num_experiments:
                                # get experiment name
                                frame_settings = [p_list for p_list in self.project_list if p_list[2] == str(i)]
                                name = frame_settings[0][0]
                                # save indexes
                                m_indexes += [i for i, p_list in enumerate(self.project_list) if p_list[2] == str(i)]
                                # check if curr experiment exists
                                if not exists(join(self.current_folder, filename, name)):
                                    # add missing experiments
                                    m_experiments.append(name + '\n')
                            if m_experiments:
                                # notify of missing experiments
                                showinfo('Project missing experiments',
                                         f"The could not find experiments:\n{m_experiments}")
                                # remove missing experiment from saved data
                                for i in m_indexes:
                                    del self.project_list[i]
                                # update .csv
                                self.__save_project_list__(trg_path)
                            # separate frames
                            if self.project_list:
                                for i in num_experiments:
                                    # get frame settings
                                    frame_settings = [p_list for p_list in self.project_list if p_list[2] == str(i)]
                                    # load from
                                    self.__load_experiment_frame__(frame_settings)
                else:
                    # clear memory from current project
                    self.project_list = []
                    self.project_frames = {}
                    # clear settings from current project
                    self.experiment_dir.set('D:\\')
                    self.show_pca_calculations.set(True)
                    self.plt_3d_view.set(False)
                    self.plt_3d_calibration.set(True)
                    self.exclude_film_name = []
                    self.project_samples = {}
                    self.domains_labels = ['losses_tangent']
                    self.first_concentration = {}
                    self.skip = {}
                    self.plot_options_dict = {}
                    self.calibrate_with_control_sample = {}
                    # select current project as new project
                    self.project_dir = filename
                    # update view
                    for widget in self.project_frame.winfo_children():
                        widget.destroy()
                    for widget in self.data_frame.winfo_children():
                        widget.destroy()
                    ttk.Radiobutton(self.project_frame, text=f" current project: {filename} ", variable=self.project_item,
                                    value=f"{{'type': 'project', 'value': '{filename}'}}",
                                    style='IndicatorOff.TRadiobutton').pack(pady=10)
                    # save project
                    mkdir(trg_path)
                    with open(join(self.current_folder, self.project_dir, self.project_dir + ".csv"), "w"):
                        pass
            else:
                showerror('Error', 'No project was created')
        # open project
        elif clicked == "open project":
            try:
                # select project
                file_path = filedialog.askdirectory(initialdir=self.current_folder, title='Select Project to open')
                if not file_path:
                    return
                filename = file_path[file_path.rfind('/') + 1:]
                if filename:
                    # def trg_path
                    trg_path = join(self.current_folder, filename, filename + ".csv")
                    # check if .csv exists
                    if not exists(trg_path):
                        showerror('Error', 'No save file from selected project was found.')
                    else:
                        # clear project_list
                        self.project_list = []
                        # clear self.project_frames
                        self.project_frames = {}
                        # select current project as chosen
                        self.project_dir = filename
                        # load settings
                        self.__load_settings__()
                        # update view
                        for widget in self.project_frame.winfo_children():
                            widget.destroy()
                        for widget in self.data_frame.winfo_children():
                            widget.destroy()
                        ttk.Radiobutton(self.project_frame, text=f" current project: {filename} ", variable=self.project_item,
                                        value=f"{{'type': 'project', 'value': '{filename}'}}",
                                        style='IndicatorOff.TRadiobutton').pack(pady=10)
                        # get project_list
                        with open(trg_path, "r", newline='') as the_file:
                            reader = csv.reader(the_file)
                            for p_list in reader:
                                self.project_list.append(p_list)
                        # get numbers of the experiments
                        num_experiments = list(
                            set([int(p_list[2]) for p_list in self.project_list]))  # list with unique identifiers
                        # def immediate experiments
                        immediate_samples = next(walk(join(self.current_folder, filename)))[1]
                        # def saved experiments
                        saved_experiments = [p_list[0] for p_list in self.project_list if
                                             eval(p_list[1])['type'] == 'experiment']
                        # check if project_list contains all immediate dirs
                        if not all([item in saved_experiments for item in immediate_samples]):
                            # identify missing experiments
                            m_experiments = [item for item in immediate_samples if item not in saved_experiments]
                            # resolve answer
                            if askquestion('Warning',
                                           "There were found experiment(s) not saved.\nWant to recover them?") == 'yes':
                                for experiment_name in m_experiments:
                                    # fill frame_settings from dir
                                    # get current identifier for project_list
                                    frame_num = str(max(num_experiments) + 1)  # frame numbers start at 0
                                    # get names
                                    samples = next(walk(join(self.current_folder, filename, experiment_name)))[1]
                                    # compound
                                    frame_settings = [[experiment_name,
                                                       f"{{'type': 'experiment', 'value': '{experiment_name}'}}",
                                                       frame_num]]
                                    for sample in samples:
                                        frame_settings.append(
                                            [sample,
                                             f"{{'type': 'sample', 'value': ('{experiment_name}', '{sample}')}}",
                                             frame_num])
                                    # add to project_list
                                    self.project_list.extend(frame_settings)
                                    # update num_experiments
                                    num_experiments.append(int(frame_num))
                                    # update csv
                                    self.__save_project_list__(trg_path)
                            else:
                                # delete unsaved folders
                                for experiment_name in m_experiments:
                                    rmtree(join(self.current_folder, filename, experiment_name))
                        # check if all immediate dirs in project_list exists
                        # def m_experiments
                        m_experiments = []
                        m_indexes = []
                        for i in num_experiments:
                            # get experiment name
                            frame_settings = [p_list for p_list in self.project_list if p_list[2] == str(i)]
                            name = frame_settings[0][0]
                            # save indexes
                            m_indexes += [i for i, p_list in enumerate(self.project_list) if p_list[2] == str(i)]
                            # check if curr experiment exists
                            if not exists(join(self.current_folder, filename, name)):
                                # add missing experiments
                                m_experiments.append(name + '\n')
                        if m_experiments:
                            # notify of missing experiments
                            showinfo('Project missing experiments', f"The could not find experiments:\n{m_experiments}")
                            # remove missing experiment from saved data
                            for i in m_indexes:
                                del self.project_list[i]
                            # update .csv
                            self.__save_project_list__(trg_path)
                        # separate frames
                        if self.project_list:
                            for i in num_experiments:
                                # get frame settings
                                frame_settings = [p_list for p_list in self.project_list if p_list[2] == str(i)]
                                # load from
                                self.__load_experiment_frame__(frame_settings)
            except Exception as error:
                showerror('Error', str(error))
                return
        # save project
        elif clicked == "save project":
            # def trg_path
            trg_path = join(self.current_folder, self.project_dir, self.project_dir + ".csv")
            # update file
            self.__save_project_list__(trg_path)
        # remove project
        elif clicked == "remove project":
            try:
                # select project
                file_path = filedialog.askdirectory(initialdir=self.current_folder, title='Select Project to remove')
                if not file_path:
                    return
                filename = file_path[file_path.rfind('/') + 1:]
                # remove folder
                if exists(file_path):
                    rmtree(file_path)
                # if removed == current
                if filename == self.project_dir:
                    # clear view
                    for widget in self.project_frame.winfo_children():
                        widget.destroy()
                    for widget in self.data_frame.winfo_children():
                        widget.destroy()
                    # set view
                    project_lbl = Label(self.project_frame, font=('Helvetica bold', 15), fg='#808080',
                                        text='No project selected')
                    project_lbl.grid(row=0, column=0, padx=40, pady=self.height / 3, sticky=N + S + W + E)
                    # clear project_list
                    self.project_list = []
                    # clear self.project_frames
                    self.project_frames = {}
                    # clear project_dir
                    self.project_dir = ''
            except Exception as error:
                showerror('Error', str(error))
                return
        # add experiment
        elif clicked == "add experiment":
            if self.project_dir == '':
                showerror('Error', 'No project was selected yet!')
            else:
                try:
                    # select file
                    file_path = filedialog.askdirectory(initialdir=self.experiment_dir.get(), title='Select Folder to add')
                    if not file_path:
                        return
                    filename = file_path[file_path.rfind('/') + 1:]
                    samples = next(walk(file_path))[1]
                    # check if it has samples
                    if not samples:
                        showerror('Error', 'experiment has no samples.')
                        return
                    if filename:
                        # add reference to project_samples
                        self.project_samples[filename] = [[samples[0]], samples[1:]]
                        # plot_options_dict
                        for num, option_key in enumerate(self.plot_options_keys):
                            key = filename + option_key
                            if num == 1 or num == 3 or num == 7:
                                self.plot_options_dict[key] = True  # default: linear, max,  pca same w&norm
                            else:
                                self.plot_options_dict[key] = False
                        # set 1st concentration
                        self.first_concentration[filename] = '9'
                        # set skip
                        self.skip[filename] = '1'
                        # set calibrate_with_control_sample
                        self.calibrate_with_control_sample[filename] = False
                        # get current identifier for project_list
                        frame_num = '0'
                        if self.project_list:
                            # get number of experiments
                            frame_num = str(
                                max([int(p_list[2]) for p_list in self.project_list]) + 1)  # frame numbers start at 0
                        # def trg_path
                        trg_path = join(self.current_folder, self.project_dir, filename)
                        # checks if filename already exists in project dir
                        if exists(trg_path):
                            if askquestion('experiment already exists', f"Replace {filename}?") == 'yes':
                                # remove experiment folder
                                rmtree(trg_path)
                                # clear old frame
                                self.project_frames[filename].destroy()
                                # clear old frame reference
                                del self.project_frames[filename]
                                # replace identifier with that from old instance
                                for p_list in self.project_list:
                                    if p_list[0] == filename:  # name == filename
                                        frame_num = p_list[2]
                                        break
                                # clear old instance from project_list
                                self.project_list = [p_list for p_list in self.project_list if p_list[2] != frame_num]
                            else:
                                # cancel operation
                                return
                        # add file to project
                        copytree(file_path, trg_path)
                        # create experiment frame
                        self.__create_experiment_frame__(filename, frame_num, samples)
                        # set curr project_item as experiment
                        self.project_item.set(filename)
                        # update .csv
                        trg_path = join(self.current_folder, self.project_dir, self.project_dir + ".csv")
                        self.__save_project_list__(trg_path)
                        # create _corrected.csv for each sample
                        for sample in samples:
                            trg_path = join(self.current_folder, self.project_dir, filename, sample)
                            s = Sample()
                            s.read_files(trg_path)
                except Exception as error:
                    showerror('Error', str(error))
        # remove experiment
        elif clicked == "remove experiment":
            try:
                # get experiment to remove
                file_path = filedialog.askdirectory(initialdir=join(self.current_folder, self.project_dir),
                                                    title='Select Folder to add')
                if not file_path:
                    return
                filename = file_path[file_path.rfind('/') + 1:]
                # remove folder
                rmtree(file_path)
                # destroy frame
                self.project_frames[filename].destroy()
                # clear frame's reference
                del self.project_frames[filename]
                # remove experiment project_samples
                del self.project_samples[filename]
                # remove experiment & samples from project_list
                frame_num = '0'
                for p_list in self.project_list:
                    if p_list[0] == filename:  # name == filename
                        frame_num = p_list[2]
                        break
                self.project_list = [p_list for p_list in self.project_list if p_list[2] != frame_num]
                # select first element of project_list as current experiment
                self.project_item.set(self.project_list[0][0])
                # update .csv
                trg_path = join(self.current_folder, self.project_dir, self.project_dir + ".csv")
                self.__save_project_list__(trg_path)
            except Exception as error:
                showerror('Error', str(error))

    @staticmethod
    def __disable_checkbox__(checkbox, bool_var):
        checkbox.configure(state=NORMAL if bool_var else DISABLED)

    def __edit_exclude_film_name__(self, name, add):
        if add:
            self.exclude_film_name.append(name)
        else:
            self.exclude_film_name.remove(name)

    def __edit_domains_labels__(self, name, add):
        if add:
            self.domains_labels.append(name)
        else:
            self.domains_labels.remove(name)

    def __edit_project_samples__(self, e_name, s_name, value):
        if value:
            # mark as control sample
            self.project_samples[e_name][0].append(s_name)
            self.project_samples[e_name][1].remove(s_name)
        else:
            # mark as experimental sample
            self.project_samples[e_name][1].append(s_name)
            self.project_samples[e_name][0].remove(s_name)

    def __save_settings__(self):
        # save config settings in config.csv
        trg_path = join(self.current_folder, self.project_dir, "settings.csv")
        with open(trg_path, "w", newline='') as the_file:
            # create vars
            writer = csv.writer(the_file)
            settings = {'experiment_dir': self.experiment_dir.get(), 'show_pca_calculations': self.show_pca_calculations.get(),
                        'plt_3d_view': self.plt_3d_view.get(), 'plt_3d_calibration': self.plt_3d_calibration.get(),
                        'calibrate_with_control_sample': self.calibrate_with_control_sample,
                        'exclude_film_name': self.exclude_film_name, 'project_samples': self.project_samples,
                        'domains_labels': self.domains_labels, 'first_concentration': self.first_concentration,
                        'skip': self.skip, 'plot_options_dict': self.plot_options_dict}
            # save settings
            for key, value in settings.items():
                writer.writerow([key, value])

    def __exit_settings__(self):
        if self.project_dir:
            # save changes to settings
            self.__save_settings__()
            # close window
            self.config_view.destroy()
            self.config_btn.configure(state=NORMAL)

    def __load_settings__(self):
        # load config settings from config.csv
        trg_path = join(self.current_folder, self.project_dir, "settings.csv")
        if not exists(trg_path):
            return
        with open(trg_path, newline='') as the_file:
            # create vars
            reader = csv.reader(the_file)
            settings = {}
            # load settings
            for key, value in reader:
                settings[key] = value
            # set configurations from settings
            self.experiment_dir.set(settings['experiment_dir'])  # only experiment_dir is a string
            self.show_pca_calculations.set(eval(settings['show_pca_calculations']))
            self.plt_3d_view.set(eval(settings['plt_3d_view']))
            self.plt_3d_calibration.set(eval(settings['plt_3d_calibration']))
            self.exclude_film_name = eval(settings['exclude_film_name'])
            self.project_samples = eval(settings['project_samples'])
            self.domains_labels = eval(settings['domains_labels'])
            self.first_concentration = eval(settings['first_concentration'])
            self.skip = eval(settings['skip'])
            self.plot_options_dict = eval(settings['plot_options_dict'])
            self.calibrate_with_control_sample = eval(settings['calibrate_with_control_sample'])

    def __edit_first_concentration__(self, sv, e_name):
        value = sv.get()
        if value:
            if value.isdigit():
                self.first_concentration[e_name] = value
            else:
                showwarning('Warning', 'Value must be a number.')
                sv.set(self.first_concentration[e_name])

    def __edit_skip__(self, sv, e_name):
        value = sv.get()
        if value:
            if value.isdigit():
                self.skip[e_name] = value
            else:
                showwarning('Warning', 'Value must be a number.')
                sv.set(self.skip[e_name])

    def __edit_calibrate_with_control_sample__(self, e_name, value):
            self.calibrate_with_control_sample[e_name] = value

    def __edit_plot_options_dict__(self, key, value):
        self.plot_options_dict[key] = value

    def __config_execute__(self):
        if self.project_dir:
            # create config_view
            self.config_view = Toplevel(self.root)
            self.config_view.title('settings')
            self.config_view.geometry(f"{int(self.width * .6)}x{int(self.height * .7)}")
            self.config_view.protocol("WM_DELETE_WINDOW", self.__exit_settings__)
            self.config_view.iconbitmap(join(self.current_folder, 'Icon.ico'))
            self.config_btn.configure(state=DISABLED)
            # create notebook
            config_notebook = ttk.Notebook(self.config_view, style='IndicatorOff.TNotebook')
            config_notebook.pack(fill="both", expand=1)
            # create project tab
            project_tab = Frame(config_notebook, relief=FLAT, width=0)
            project_tab.pack(fill="both", expand=1)
            config_notebook.add(project_tab, text='project')
            project_tab.columnconfigure(0, weight=1, minsize=75)
            project_tab.rowconfigure(0, weight=1, minsize=75)
            project_tab.rowconfigure(1, weight=4)
            # fill project tab
            work_folder_frame = Frame(project_tab)
            work_folder_frame.grid(row=0, column=0, sticky=N + W + S + E)
            work_folder_frame.columnconfigure(1, weight=1, minsize=75)
            work_folder_frame.rowconfigure(0, weight=1, minsize=75)
            Label(work_folder_frame, text="set work folder:").grid(row=0, column=0, padx=10)
            Entry(work_folder_frame, textvariable=self.experiment_dir).grid(row=0, column=1, sticky=W + E, padx=(0, 30))
            checklist_frame = Frame(project_tab)
            checklist_frame.grid(row=1, column=0, sticky=N + W + S + E)
            c_pca_calc = Checkbutton(checklist_frame, text="show pca calculations", variable=self.show_pca_calculations,
                                     onvalue=True, offvalue=False)
            c_calibration = Checkbutton(checklist_frame, text="show 3d calibration", variable=self.plt_3d_calibration,
                                        onvalue=True, offvalue=False, state=NORMAL if self.plt_3d_view.get() else DISABLED)
            c_3d_view = Checkbutton(checklist_frame, text="show 3d view", variable=self.plt_3d_view,
                                    onvalue=True, offvalue=False,
                                    command=lambda: self.__disable_checkbox__(c_calibration, self.plt_3d_view.get()))
            domains_labels_lbl = Label(checklist_frame, text="Select parameters to use in PCA:")
            exclude_film_lbl = Label(checklist_frame, text="Select films to ignore:")
            domains_labels_frame = Frame(checklist_frame, padx=50, pady=50, borderwidth=3,
                                         highlightbackground="#999999", highlightthickness=2)
            exclude_film_frame = Frame(checklist_frame, padx=50, pady=50, borderwidth=3, highlightbackground="#999999",
                                       highlightthickness=2)
            c_pca_calc.grid(row=0, column=0)
            c_3d_view.grid(row=0, column=1)
            c_calibration.grid(row=0, column=2)
            num_checklist_frame_columns = checklist_frame.grid_size()[0]
            domains_labels_lbl.grid(row=1, column=0, padx=10, pady=(20, 0), sticky=W,
                                    columnspan=num_checklist_frame_columns)
            domains_labels_frame.grid(row=2, column=0, columnspan=num_checklist_frame_columns,
                                      pady=10, padx=15, sticky=N + W + S + E)
            checklist_frame.rowconfigure(2, weight=1, minsize=20)
            exclude_film_lbl.grid(row=3, column=0, padx=10, pady=(0, 0), sticky=W,
                                  columnspan=num_checklist_frame_columns)
            exclude_film_frame.grid(row=4, column=0, columnspan=num_checklist_frame_columns,
                                    pady=10, padx=15, sticky=N + W + S + E)
            checklist_frame.rowconfigure(4, weight=1, minsize=75)
            for i in range(num_checklist_frame_columns):
                checklist_frame.columnconfigure(i, weight=1, minsize=75)
            # fill project tab: domains_labels_frame
            num_domains_labels_frame_columns = 3
            possible_domains_labels = ['impedance', 'phase', 'real_impedance', 'imaginary_impedance', 'losses_tangent']
            for i in range(num_domains_labels_frame_columns):
                domains_labels_frame.columnconfigure(i, weight=1, minsize=20)
            domains_labels_var_dict = {}
            for column_count, domain in enumerate(possible_domains_labels):
                # create temp bool var
                v = BooleanVar()
                v.set(domain in self.domains_labels)
                # save temp var in higher scope
                domains_labels_var_dict[domain] = v
                # create checkbox & save tmp domain value in fixed name var
                c = Checkbutton(domains_labels_frame, text=domain, variable=v, onvalue=True, offvalue=False,
                                command=lambda name=domain: self.__edit_domains_labels__(name, domains_labels_var_dict[name].get()))
                # show checkbox
                c.grid(row=int(column_count / num_domains_labels_frame_columns),
                       column=int(column_count % num_domains_labels_frame_columns), sticky=W + N + S)
            num_domains_labels_frame_rows = domains_labels_frame.grid_size()[1]
            for i in range(num_domains_labels_frame_rows):
                domains_labels_frame.rowconfigure(i, weight=1, minsize=20)
            # fill project tab: exclude_film_frame
            num_exclude_film_frame_columns = 4
            for i in range(num_exclude_film_frame_columns):
                exclude_film_frame.columnconfigure(i, weight=1, minsize=20)
            exclude_film_var_dict = {}
            for column_count, key in enumerate(self.project_samples):
                # create temp bool var
                v = BooleanVar()
                v.set(key in self.exclude_film_name)
                # save temp var in higher scope
                exclude_film_var_dict[key] = v
                # create checkbox & save tmp key value in fixed name var
                c = Checkbutton(exclude_film_frame, text=key, variable=v, onvalue=True, offvalue=False,
                                command=lambda name=key: self.__edit_exclude_film_name__(name, exclude_film_var_dict[name].get()))
                # show checkbox
                c.grid(row=int(column_count / num_exclude_film_frame_columns),
                       column=int(column_count % num_exclude_film_frame_columns), sticky=W + N + S)
            num_exclude_film_frame_rows = exclude_film_frame.grid_size()[1]
            for i in range(num_exclude_film_frame_rows):
                exclude_film_frame.rowconfigure(i, weight=1, minsize=20)
            # create experiment tabs
            control_samples_dict = {}
            first_concentration_var_dict = {}
            skip_var_dict = {}
            calibrate_with_control_sample_var_dict = {}
            plot_options_vars_dict = {}
            for experiment_name, value in self.project_samples.items():
                # create experiment tab
                frame = Frame(config_notebook, width=0)
                frame.pack(fill="both", expand=1)
                config_notebook.add(frame, text=experiment_name)
                frame.columnconfigure(0, weight=1, minsize=150)
                frame.columnconfigure(1, weight=1, minsize=50)
                frame.columnconfigure(2, weight=1, minsize=175)
                frame.columnconfigure(3, weight=1, minsize=50)
                frame.columnconfigure(4, weight=1, minsize=250)
                frame.rowconfigure(0, weight=1, minsize=75)
                frame.rowconfigure(1, weight=8, minsize=75)
                # fill experiment tab: 1st concentration
                experiment_first_concentration = StringVar()
                experiment_first_concentration.trace("w", lambda name, index, mode, sv=experiment_first_concentration, e_name=experiment_name: self.__edit_first_concentration__(sv, e_name))
                experiment_first_concentration.set(self.first_concentration[experiment_name])
                first_concentration_var_dict[experiment_name] = experiment_first_concentration
                Label(frame, text="-log 1st concentration:"). \
                    grid(row=0, column=0, padx=10)
                Entry(frame, textvariable=first_concentration_var_dict[experiment_name]). \
                    grid(row=0, column=1, sticky=W + E, padx=(0, 30))
                # fill experiment tab: skip
                experiment_skip = StringVar()
                experiment_skip.trace("w", lambda name, index, mode, sv=experiment_skip, e_name=experiment_name: self.__edit_skip__(sv, e_name))
                experiment_skip.set(self.skip[experiment_name])
                skip_var_dict[experiment_name] = experiment_skip
                Label(frame, text="gap between concentrations:"). \
                    grid(row=0, column=2, padx=10)
                Entry(frame, textvariable=skip_var_dict[experiment_name]). \
                    grid(row=0, column=3, sticky=W + E, padx=(0, 30))
                # fill experiment tab: calibrate_with_control_sample
                experiment_calibrate_with_control_sample = BooleanVar()
                experiment_calibrate_with_control_sample.set(self.calibrate_with_control_sample[experiment_name])
                calibrate_with_control_sample_var_dict[experiment_name] = experiment_calibrate_with_control_sample
                Checkbutton(frame, text="calibrate with control sample",
                            variable=experiment_calibrate_with_control_sample, onvalue=True, offvalue=False,
                            command=lambda e_name=experiment_name: self.__edit_calibrate_with_control_sample__(e_name, calibrate_with_control_sample_var_dict[e_name].get()))\
                    .grid(row=0, column=4, columnspan=1, sticky=W + E, padx=(0, 60))
                # fill experiment tab: create checklist frame
                checklist_frame = Frame(frame)
                checklist_frame.grid(row=1, column=0, columnspan=5, sticky=N + W + S + E)
                # fill experiment tab: select plots to show
                plot_options_frame = Frame(checklist_frame)
                plot_options_frame.pack(fill=BOTH, expand=1)
                Label(plot_options_frame, text="select plots to show:"). \
                    grid(row=0, column=0, padx=10, pady=20, columnspan=4, sticky=W + N)
                num_plot_options_frame_columns = 4
                for num, option_key in enumerate(self.plot_options_keys):
                    key = experiment_name + option_key
                    v = BooleanVar()
                    if key in self.plot_options_dict.keys():
                        v.set(self.plot_options_dict[key])
                    else:
                        v.set(False)
                    plot_options_vars_dict[key] = v
                    Checkbutton(plot_options_frame, text=option_key[1:].replace('_', ' '),
                                variable=v, onvalue=True, offvalue=False,
                                command=lambda ref=key: self.__edit_plot_options_dict__(ref, plot_options_vars_dict[ref].get())). \
                        grid(row=1 + int(num / num_plot_options_frame_columns),
                             column=int(num % num_plot_options_frame_columns), pady=10)
                for i in range(4):
                    plot_options_frame.columnconfigure(i, weight=1, minsize=20)
                # fill experiment tab: control samples
                num_control_samples_frame_columns = 4
                control_samples_frame = Frame(checklist_frame)
                control_samples_frame.pack(fill=BOTH, expand=1)
                samples = Project.sorted_alphanumeric(value[0] + value[1])  # sorting keeps the order showed independent
                num_experiments = len(samples)
                Label(control_samples_frame, text="select control samples:"). \
                    grid(row=0, column=0, padx=10, pady=20, columnspan=num_experiments, sticky=W + N)
                for num, sample in enumerate(samples):
                    v = BooleanVar()
                    v.set(sample in value[0])
                    key = experiment_name + '_' + sample
                    control_samples_dict[key] = v
                    sample_checkbox = Checkbutton(control_samples_frame, text=sample, variable=v, onvalue=True,
                                                  offvalue=False,
                                                  command=lambda e_name=experiment_name, s_name=sample, ref=key:
                                                  self.__edit_project_samples__(e_name, s_name, control_samples_dict[ref].get()))
                    sample_checkbox.grid(row=int(1 + num / num_control_samples_frame_columns),
                                         column=int(num % num_control_samples_frame_columns), sticky=N)
                num_columns, num_rows = control_samples_frame.grid_size()
                for i in range(num_columns):
                    control_samples_frame.columnconfigure(i, weight=1, minsize=20)
                for i in range(1, num_rows):
                    control_samples_frame.rowconfigure(i, weight=1, minsize=20)

    def __help_execute__(self, clicked):
        # reset help dropdown
        self.help_clicked.set("Help")
        # build help window
        help_popup = Toplevel(self.root)
        help_popup.title('help')
        if clicked == "how is PCA build":
            help_popup.geometry("1000x300")
            Label(help_popup,
                  text="PCA structure:\n\tlines)\t<number of concentrations> x <number of loops> x <number of experiment samples>\n\tcolumns)\t<number of frequency points> x <number parameters> x  ( <number of films> + <number of control samples> )",
                  anchor="w", justify="left", font=13).pack(fill=X, padx=20, pady=10, expand=True)
            Label(help_popup,
                  text="PCA uses:\n\t1) minimum number of experiment samples\n\t2) minimum number of loops\n\t3) minimum set of concentrations",
                  anchor="w", justify="left", font=13).pack(fill=X, padx=20, pady=10, expand=True)
        if clicked == "what can cause errors":
            help_popup.geometry("800x300")
            Label(help_popup,
                  text="1) *names from min set of concentrations must be present in all other samples from used experiments\n2) same freq within experiment\n* names from scv must be <indicator>.csv or <indicator>_<...>.csv",
                  anchor="w", justify="left", font=13).pack(fill=X, padx=20, pady=10, expand=True)

    def __run_execute__(self):
        if self.project_dir:
            # get dict
            d = self.project_item.get()
            if d:
                d = eval(d)
            else:
                return
            # clear data vars
            self.data_images = []
            # clear view
            for i, widget in enumerate(self.data_frame.winfo_children()):
                self.data_frame.rowconfigure(i, minsize=0, weight=1)  # reset grid minsize
                widget.destroy()  # destroy image
            self.data_frame_scroll.reset_scroll()
            # determine type of script
            t = d['type']
            try:
                if t == 'project':
                    # get vars
                    project_name = d['value']
                    # check if domains_labels is empty
                    if not self.domains_labels:
                        showerror('Error', "No domains were selected.")
                        return
                    # create Project var
                    experiment_list = []
                    for experiment in self.project_samples:
                        experiment_list.append([int(self.first_concentration[experiment]), int(self.skip[experiment])])
                    experiments = list(self.project_samples.keys())
                    for e_name in self.exclude_film_name:
                        if e_name in experiments:
                            experiments.remove(e_name)
                    p = Project(project_name=project_name, domains_labels=self.domains_labels,
                                experiment_list=experiment_list,
                                experiments=experiments, project_samples=self.project_samples,
                                folder=join(self.current_folder, project_name))
                    # get images
                    p.create_heatmap()
                    images = [join(self.current_folder, project_name, 'pca.png')]
                    if self.show_pca_calculations.get():
                        weights_images = Project.show_weights(p)
                        images.extend(weights_images)
                    if self.plt_3d_view.get():
                        images.append(p.create_3d_view(self.plt_3d_calibration.get()))
                elif t == 'experiment':
                    # get vars
                    experiment_name = d['value']
                    trg_path = join(self.current_folder, self.project_dir, experiment_name)
                    # create Experiment
                    e = Experiment(samples=self.project_samples[experiment_name],
                                   first_concentration=int(self.first_concentration[experiment_name]),
                                   skip=int(self.skip[experiment_name]), path=trg_path,
                                   calibrate_with_control_sample=self.calibrate_with_control_sample[experiment_name])
                    e.get_values()
                    # get images
                    images = e.show_results(show_all=self.plot_options_dict[experiment_name + '_show_all'],
                                            show_domains=self.plot_options_dict[experiment_name + '_show_linear_scale'],
                                            show_db=self.plot_options_dict[experiment_name + '_show_db'],
                                            show_normal_pc1=self.plot_options_dict[experiment_name + '_show_normal_pc1'],
                                            show_max_variance=self.plot_options_dict[
                                                experiment_name + '_show_max_variance'],
                                            show_pc1_sample1_weight=self.plot_options_dict[
                                                experiment_name + '_show_pc1_sample1_weight'],
                                            show_pc1_sample1_normalization=self.plot_options_dict[
                                                experiment_name + '_show_pc1_sample1_normalization'],
                                            show_pc1_sample1_weight_normalization=self.plot_options_dict[
                                                experiment_name + '_show_pc1_sample1_weight_normalization'], )
                elif t == 'sample':
                    # get vars
                    experiment_name, sample_name = d['value']
                    trg_path = join(self.current_folder, self.project_dir, experiment_name, sample_name)
                    # create Sample
                    s = Sample(plt_log_scale=self.plot_options_dict[experiment_name + '_show_db'] or self.plot_options_dict[
                        experiment_name + '_show_all'])
                    s.read_files(trg_path)
                    # get images
                    images = s.results(impedance=True, phase=True, r_impedance=True, i_impedance=True)
                else:
                    showerror('Error', 'Save file was corrupted.')
                    return
                # save images
                self.last_images = images
                # create view
                for i, img in enumerate(images):
                    # create image
                    image = ImageTk.PhotoImage(Image.open(img))
                    # save image in global var
                    self.data_images.append(image)
                    # show image
                    Label(self.data_frame, image=image).grid(row=i, column=0, pady=10, sticky=W + E)
                # adjust self.data_frame size
                num_images = len(self.data_images)
                total_images_height = self.data_images[0].height() * num_images
                if not total_images_height > self.height - 110:
                    # images need to be padded to fit data frame (if not, it gets decentralized and activates bad scroll)
                    h = int(round((self.height - 110) / num_images))
                    for i in range(num_images):
                        self.data_frame.rowconfigure(i, minsize=h, weight=1)
            except Exception as error:
                showerror('Error', str(error))

    def __close_program__(self):
        # save settings in case settings menu was still open
        if self.project_dir:
            self.__save_settings__()
        # make sure program closes after root is closed
        sys.exit()

    def __init__(self, cwd=getcwd()):
        # get ETDA_Projects path
        current_folder = join(cwd, "ETDA_Projects")
        if not exists(current_folder):
            showerror('Error', 'Program was corrupted.\nPlease reinstall it.')
        self.current_folder = current_folder
        # def root
        root = Tk()
        root.title('ETDA')
        width = root.winfo_screenwidth()
        self.width = width
        height = root.winfo_screenheight()
        self.height = height
        root.state('zoomed')
        root.protocol("WM_DELETE_WINDOW", self.__close_program__)
        root.iconbitmap(default=join(current_folder, 'Icon.ico'))
        self.root = root
        experiment_dir = StringVar()
        experiment_dir.set('D:\\')
        self.experiment_dir = experiment_dir

        # def style
        style = ttk.Style(root)
        style.theme_use('classic')  # 'aqua', 'step', 'clam', 'alt', 'default', 'classic'
        style.configure('IndicatorOff.TRadiobutton', indicatorrelief=FLAT,
                        indicatormargin=-1, indicatordiameter=-1, relief=FLAT,
                        focusthickness=0, highlightthickness=0, padding=5)
        style.map('IndicatorOff.TRadiobutton', background=[('selected', 'white'), ('active', '#ececec')])
        style.configure('IndicatorOff.TNotebook', padding=10)
        self.style = style

        # def menu vars
        self.last_images = []
        self.project_dir = ''
        self.exclude_film_name = []
        self.domains_labels = ['losses_tangent']  # possible domains_labels = {'impedance', 'phase', 'real_impedance', 'imaginary_impedance', 'losses_tangent'}
        plt_3d_view = BooleanVar()
        plt_3d_view.set(False)
        self.plt_3d_view = plt_3d_view
        plt_3d_calibration = BooleanVar()
        plt_3d_calibration.set(True)
        self.plt_3d_calibration = plt_3d_calibration
        show_pca_calculations = BooleanVar()
        show_pca_calculations.set(True)
        self.show_pca_calculations = show_pca_calculations

        self.first_concentration = {}
        self.skip = {}
        self.calibrate_with_control_sample = {}
        self.project_samples = {}
        self.plot_options_dict = {}
        self.plot_options_keys = ['_show_all', '_show_linear_scale', '_show_db', '_show_max_variance', '_show_normal_pc1',
                                  '_show_pc1_sample1_weight', '_show_pc1_sample1_normalization',
                                  '_show_pc1_sample1_weight_normalization']

        file_options = [  # CRUD
            "new project",
            "open project",
            "save project",
            "remove project",
            "add experiment",
            "remove experiment"
        ]
        self.file_clicked = StringVar()
        self.file_clicked.set("File")

        help_options = [  # ask for feedback
            "how is PCA build",
            "what can cause errors"
        ]
        self.help_clicked = StringVar()
        self.help_clicked.set("Help")

        # def project vars
        self.project_item = StringVar()
        self.project_list = []  # [[text, value, frame_num], ...]
        self.project_frames = {}  # {<experiment value>:<frame ref>, ...}

        # def frames
        menu_frame = LabelFrame(root, height=30, width=width, bd=1, relief=SUNKEN)
        # setup scrolls
        self.project_frame_base = Frame(root, height=height - 110, width=width / 5, borderwidth=3, relief=GROOVE)
        self.data_frame_base = Frame(root, height=height - 110, width=width * 4 / 6)

        # config frames items
        menu_frame.grid_propagate(False)
        # self.project_frame_base.grid_propagate(False)
        self.project_frame_base.pack_propagate(False)
        self.data_frame_base.grid_propagate(False)
        self.data_frame_base.pack_propagate(False)
        # show frames
        menu_frame.grid(row=0, column=0, columnspan=2, sticky=N + W + S + E)
        self.project_frame_base.grid(row=1, column=0, padx=5, pady=10, sticky=N + W + S + E)
        self.data_frame_base.grid(row=1, column=1, padx=5, pady=10, sticky=N + W + S + E)

        # ScrolledFrames
        self.project_frame_scroll = ScrolledFrame(self.project_frame_base)
        self.data_frame_scroll = ScrolledFrame(self.data_frame_base)

        # final frames
        self.project_frame = self.project_frame_scroll.get_frame()
        self.data_frame = self.data_frame_scroll.get_frame()

        # create menu items
        file_drop = OptionMenu(menu_frame, self.file_clicked, *file_options, command=self.__file_execute__)
        self.config_btn = Button(menu_frame, text="settings", command=self.__config_execute__, relief=FLAT)
        help_drop = OptionMenu(menu_frame, self.help_clicked, *help_options, command=self.__help_execute__)
        run_btn = Button(menu_frame, text="run", command=self.__run_execute__)
        export_btn = Button(menu_frame, text="export images", command=self.__export_images__)
        # config menu items
        file_drop.config(width=5, borderwidth=0, indicatoron=False)
        help_drop.config(width=5, borderwidth=0, indicatoron=False)
        # show menu items
        file_drop.grid(row=0, column=0, padx=1, sticky=W)
        self.config_btn.grid(row=0, column=1, padx=1, sticky=W)
        help_drop.grid(row=0, column=2, padx=1, sticky=W)
        run_btn.grid(row=0, column=3, padx=20, sticky=W)
        export_btn.grid(row=0, column=4, padx=20, sticky=W)

        # create project items
        project_lbl = Label(self.project_frame, font=('Helvetica bold', 15), fg='#808080', text="No project selected")
        # show project items
        project_lbl.grid(row=0, column=0, padx=40, pady=height / 3, sticky=N + S + W + E)

        # def icon img
        image = Image.open(join(current_folder, '9500.png'))
        image = image.resize((30, 30))
        self.img_u252x0 = ImageTk.PhotoImage(image)
        image = Image.open(join(current_folder, '9492.png'))
        image = image.resize((30, 30))
        self.img_u251x4 = ImageTk.PhotoImage(image)

        root.mainloop()
