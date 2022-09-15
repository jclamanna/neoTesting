# builtins
import yaml, os, subprocess
from pprint import pprint
from copy import deepcopy
# vis
import matplotlib.pyplot as plt
import pylab
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
from mi3gpu.utils.seqtools import histsim
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

# data
import pandas as pd
import numpy as np 

class Vis:

    def __init__(self, seqConfig, visConfig, metricsConfig, paths, name, args=None):

        self.seqConfig = seqConfig
        self.metricsConfig = metricsConfig
        self.visConfig = visConfig
        self.paths = paths
        self.name = name
       
    def make_hams(self):
        print("\t\tmaking hams")
        # if self.loglog:
        #     print("\t\t\tplotting loglog")
        #     self.plot_hams_loglog()
        self.plot_hams()
        print("\t\tcompleted: plotting hams")

    def make_covars(self):
        
        bvms_dir = os.listdir(self.paths["bvms_path"])
        covars_dir = os.listdir(self.paths["covars_path"])
        print("\t\tmaking covars")
        
        print("computing all bvms")
        for label in self.seqConfig.keys():
            if self.seqConfig[label]['run_model']:
                if label.lower() == "svae":
                    self.get_bvms(label, self.paths["out_path"] + 'gen_{}'.format(self.name))
                elif label.lower() == "indep" and "bvms_Indep.npy" not in bvms_dir:
                    self.get_bvms(label, self.paths["data_path"] + "indep_seqs")
                elif "bvms_"+label+".npy" in bvms_dir:
                    continue
                else:
                    self.get_bvms(label, self.paths["data_path"] + label.lower() + "_" + self.metricsConfig["sequences_name_pattern"]) 
        
        print("computing all covars")
        for label in self.seqConfig.keys():
            if self.seqConfig[label]['run_model']:
                if label.lower() == "svae":
                    self.get_covars(label, self.paths["out_path"] + "bvms_" + self.name + ".npy")
                elif "covars_"+label+".npy" in covars_dir:
                    continue
                else: 
                    self.get_covars(label, self.paths["bvms_path"] + "bvms_" + label + ".npy") 
    
        print("\n\t\tplotting covars")
        self.plot_covars()
        print("\t\tcompleted: making covars")

    def make_r20(self):
        self.makeDB()
        self.compute()
        self.plotr20()

import Hams
import Covars
import r20
