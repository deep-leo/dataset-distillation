import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import platform
import seaborn as sns
import torch


class Entry:
    """Notebook entry.
    """

    def __init__(self):
        self.global_steps = []
        self.items = []

    def __iter__(self):
        self.iter = zip(self.global_steps, self.items)
        return self

    def __next__(self):
        return next(self.iter)

    def __len__(self):
        return len(self.global_steps)

    def add(self, item, global_step):
        """Add item to entry, indexed by global_step.
        """
        self.items.append(item)
        self.global_steps.append(global_step)


class Notebook:
    """Class for recording experiment results using a tensorboard-like API.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.data = {}
        self.fig_counter = 0

        if platform.system() == 'Linux':
            matplotlib.use('agg')

    def add_scalar(self, tag, scalar, global_step):
        if tag not in self.data:
            self.data[tag] = Entry()
        self.data[tag].add(scalar, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        for tag, scalar in tag_scalar_dict.items():
            joined_tag = main_tag + '_' + tag
            if joined_tag not in self.data:
                self.data[joined_tag] = Entry()
            self.data[joined_tag].add(scalar, global_step)

    def pandas(self):
        """Converts Notebook object to a pandas DataFrame.
        """
        raise NotImplementedError()

    def lineplot(self, tag, save=False):
        """Creates a global_step vs. tag line plot using matplotlib.
        """
        if tag not in self.data:
            raise ValueError('Tag not found.')
        plt.figure(self.fig_counter)
        self.fig_counter += 1
        plt.plot(self.data[tag].global_steps, self.data[tag].items)
        plt.xlabel('global_step')
        plt.ylabel(tag)
        if save:
            plt.savefig(os.path.join(self.save_dir, f'{tag}-lineplot.png'))

    def linesplot(self, main_tag, save=False):
        """Creates a global_step vs. main_tag line plot with multiple labeled
        lines using matplotlib.
        """
        plt.figure(self.fig_counter)
        self.fig_counter += 1
        for tag, entry in self.data.items():
            if tag.startswith(main_tag + '_'):
                plt.plot(entry.global_steps, entry.items, label=tag)
        plt.xlabel('global_step')
        plt.ylabel(main_tag)
        plt.legend()
        if save:
            plt.savefig(os.path.join(self.save_dir, f'{main_tag}-linesplot.png'))

    def relplot(self, main_tag, save=False):
        """Creates a global_step vs. main_tag relplot using seaborn.
        """
        df_data = {'global_step': [], main_tag: [], main_tag + ' tags': []}
        for tag, entry in self.data.items():
            if tag.startswith(main_tag + '_'):
                df_data['global_step'].extend(entry.global_steps)
                df_data[main_tag].extend(entry.items)
                df_data[main_tag + ' tags'].extend(len(entry) * [tag.split('_')[-1]])
        df = pd.DataFrame(df_data)

        plt.figure(self.fig_counter)
        self.fig_counter += 1
        sns.set()
        sns.relplot(x='global_step', y=main_tag, kind='line', data=df)
        if save:
            plt.savefig(os.path.join(self.save_dir, f'{main_tag}-relplot.png'))

    def save_tar(self):
        torch.save(self.data, os.path.join(self.save_dir, 'notebook.tar'))

    def load_tar(self, path):
        self.data = torch.load(path)