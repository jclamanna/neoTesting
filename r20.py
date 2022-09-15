import subprocess
import matplotlib.pyplot as plt
import pylab
import numpy as np
from Vis import Vis

class r20(Vis):

    def __init__(self, seqConf, visConf, metConf, paths, name):
        super().__init__(seqConf,visConf,metConf, paths, name)

    def makeDB(self):
        command= "./r20/HOM_r20.py make_db " + self.paths["r20_path"] + self.name + " " + self.metricsConfig["r20_pss"] + " " + self.paths["data_path"]+ "val_" + self.metricsConfig["sequences_name_pattern"] + " --npos " + self.metricsConfig['r20_start'] + "-" + self.metricsConfig['r20_end']
        subprocess.run(command, shell=True)

    def compute(self):
        command= "./r20/HOM_r20.py count " + self.paths["r20_path"] + self.name + " " + self.paths["r20_path"] + "r20_mi " + self.paths["data_path"] + "target_" + self.metricsConfig["sequences_name_pattern"] + " " + self.paths["out_path"] + "gen_" + self.name + " " + self.paths["data_path"] + "indep_seqs" 
        subprocess.run(command, shell=True)

    def plotr20(self):
        fig, ax = pylab.subplots(figsize=(3,3))
        box_font = 1.5
        #xt = np.arange(2,8,1)

        r20_output = np.load(self.paths["r20_path"] + "r20_mi.npy") 
        r20 = np.nanmean(r20_output, axis=1)
        print(r20)
        #pylab.xticks(np.arange(2,7,1))
        ax.plot(range(2,8), r20[:,0], linestyle='solid', linewidth=2, alpha=1.0, color='black', marker = 'o', label='val', zorder=15)
        ax.plot(range(2,8), r20[:,1], linestyle='solid', linewidth=2, alpha=1.0, color='orange', marker = 'o', label='model', zorder=-5)
        ax.plot(range(2,8), r20[:,2], linestyle='solid', linewidth=2, alpha=1.0, color='silver', marker = 'o', label='indep', zorder=-10)
        pylab.ylabel('r20', fontsize=11)
        pylab.xlabel('High Order Marginals', fontsize=11)
        pylab.tick_params(direction='in',axis='both', which='major', labelsize=9, length=4, width=0.6)
        ax.set_xticks(np.arange(2, 8, 1))
        pylab.tight_layout()
        pylab.legend(fontsize=7, loc="lower left")
        if self.visConfig["show_hp"]:
            ax.set_title(self.name, fontsize = 10)
        pylab.savefig(self.paths["vis_path"] + "/HOMr20_" + self.name + ".pdf", dpi=500, format='pdf', bbox_inches='tight')
        pylab.close()
