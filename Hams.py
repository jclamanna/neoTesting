import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from Vis import Vis
from os import listdir
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
from mi3gpu.utils.seqtools import histsim

class Hams(Vis):

    def __init__(self, seqConf, visConf, metConf, paths, name):
        super().__init__(seqConf,visConf,metConf, paths, name)

    
    def plot_hams(self):
        print("\n\nmaking normal hams")
        fig, ax = pylab.subplots(figsize=(self.visConfig['fig_size'], self.visConfig['fig_size']))
        box_font = self.visConfig['box_font_size']

        # axes labels
        xlabel = r"$d$"
        ylabel = "f"
        if self.metricsConfig['protein'] == "Kinase":
            start = 120
            end = 230
            x_tick_range = np.arange(start, end, 20)
            pylab.xlim(start, end)
            
        all_freqs = dict()
        msa_dir = listdir("data/")

        for label in self.seqConfig.keys():
            if not self.seqConfig[label]['run_model']:
                print("skipping ", label)
                continue
            # if not self.metricsConfig['which_models'][label]:    # model is 'false' in the which_models{}, then continue
            #     continue
            # label = self.metricsConfig['label_dict'][label]
            print("computing hams for:\t", label)

            for msa in msa_dir:
                if label.lower() == "svae":
                    seqs = loadSeqs(self.paths["out_path"] + 'gen_{}'.format(self.name), names=self.metricsConfig['ALPHA'])[0][0:self.metricsConfig['keep_hams']]
                elif label.lower() in msa.lower() and ".csv" not in msa.lower():
                    seqs = loadSeqs(self.paths["data_path"] + msa, names=self.metricsConfig['ALPHA'])[0][0:self.metricsConfig['keep_hams']]
            # seqs = loadSeqs(self.metricsConfig['msa_dir'] + "/" + seqs_file, names=self.metricsConfig['ALPHA'])[0][0:self.metricsConfig['keep_hams']]
            h = histsim(seqs).astype(float)
            h = h/np.sum(h)
            all_freqs[label] = h
            rev_h = h[::-1]
            line_style = "solid"
            if label == "Test":
                line_style = "dashed"
                ax.plot(rev_h, linestyle=line_style, linewidth=self.visConfig['line_width'],
                    alpha=1.0, color=self.seqConfig[label]['color_set'], label=label, zorder=self.seqConfig[label]['z_order'])
            else:
                ax.plot(rev_h, linestyle=line_style, linewidth=self.visConfig['line_width'],
                    alpha=self.visConfig['line_alpha'], color=self.seqConfig[label]['color_set'], label=label, zorder=self.seqConfig[label]['z_order'])

        y_tick_range = np.arange(0.0, 0.08, 0.02)
        pylab.ylabel(ylabel, fontsize=self.visConfig['label_size'])
        pylab.xlabel(xlabel, fontsize=self.visConfig['label_size'])
        x_tick_range = np.arange(0, 201, 20)
        pylab.xticks(x_tick_range, rotation=45)
        pylab.yticks(y_tick_range)
        pylab.tick_params(direction='in',axis='both', which='major', labelsize=self.visConfig['tick_size'], 
            length=self.visConfig['tick_length'], width=self.visConfig['tick_width'])
        #my_title = "Hamming Distance Distributions\n" + self.parent_dir_name
        file_name = "ham_" + self.name + ".pdf"
        #pylab.title(self.which_size, fontsize=self.title_size)
        pylab.tight_layout()
        pylab.legend(fontsize=self.visConfig['tick_size']-2, loc="upper left")
        save_name = self.paths['vis_path'] + file_name
        if self.visConfig["show_hp"]:
            ax.set_title(self.name, fontsize = 12)
        pylab.savefig(save_name, dpi=self.visConfig['dpi'], format='pdf', bbox_inches='tight')
        pylab.close()
    