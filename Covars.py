import numpy as np
import pandas as pd
import os
from mi3gpu.utils.seqload import loadSeqs
import matplotlib.pyplot as plt
import argparse
import sys
from Vis import Vis
from scipy.stats import pearsonr, spearmanr
import pylab



class Covars(Vis):

    def __init__(self, seqConf, visConf, metConf, paths, name):
        super().__init__(seqConf,visConf,metConf, paths, name)
        

    def get_bvms(self, label, msa_file):

        if not self.seqConfig[label]['run_model']:
            print("\tskipping bvms for: ", label)
        else:    
            print("bvms for: ", label) 
            print(msa_file)
            if label.lower() == "svae":
                save_name = self.paths["out_path"] + "bvms_" + self.name 
            else:
                save_name = self.paths["bvms_path"] + "bvms_" + label

            msa = loadSeqs(msa_file)[0][:self.metricsConfig['keep_covars']]
            output = self.compute_bvms(msa, self.metricsConfig['A'], '0')
            np.save(save_name, output)
            print("\t\t\t\tfinished computing bvms for:\t", label)
            print("output saved to:" + save_name)
    
    def compute_bvms(self, seqs, q, weights, nrmlz=True):
        nSeq, L = seqs.shape

        #if weights != '0':
        #    weights = np.load(weights)

        if weights == '0':
            weights = None
        
        if q > 16: # the x + q*y operation below may overflow for u1
            seqs = seqs.astype('i4')

        if nrmlz:
            nrmlz = lambda x: x/np.sum(x, axis=-1, keepdims=True)
        else:
            nrmlz = lambda x: x

        def freqs(s, bins):
            return np.bincount(s, minlength=bins, weights=weights)

            #f = nrmlz(np.array([freqs(seqs[:,i], q) for i in range(L)]))
        ff = nrmlz(np.array([freqs(seqs[:,j] + q*seqs[:,i], q*q) \
            for i in range(L-1) for j in range(i+1, L)]))
        return ff

    def get_covars(self, label, bvms_file):
        print("covars for: ", label)

        if label.lower() == "svae":
            save_name = self.paths["out_path"] + "covars_" + self.name 
        else:
            save_name = self.paths["covars_path"] + "covars_" + label

        print(bvms_file)
        bimarg = np.load(bvms_file)
        C = bimarg - self.indepF(bimarg)
        np.save(save_name, C)
        
        print("completed covars for:\t" + label)

    def getL(self, size):
        return int(((1+np.sqrt(1+8*size))//2) + 0.5)

    def getLq(self, J):
        return self.getL(J.shape[0]), int(np.sqrt(J.shape[1]) + 0.5)

    def getUnimarg(self, ff):
        L, q = getLq(ff)
        ff = ff.reshape((L*(L-1)//2, q, q))
        marg = np.array([np.sum(ff[0], axis=1)] +
                        [np.sum(ff[n], axis=0) for n in range(L-1)])
        return marg/(np.sum(marg, axis=1)[:,None]) # correct any fp errors

    def indepF(self,fab):
        L, q = self.getLq(fab)
        fabx = fab.reshape((fab.shape[0], q, q))
        fa1, fb2 = np.sum(fabx,axis=2), np.sum(fabx,axis=1)
        fafb = np.array([np.outer(fa, fb).flatten() for fa,fb in zip(fa1, fb2)])
        return fafb

    def getM(self, x, diag_fill=0):
        L = self.getL(len(x))
        M = np.empty((L,L))
        M[np.triu_indices(L,k=1)] = x
        M = M + M.T
        M[np.diag_indices(L)] = diag_fill
        return M

    def plot_covars(self):
        print("\t\t\t\tplotting covars:\t")
        fig, ax = plt.subplots(figsize=(self.visConfig['fig_size'],self.visConfig['fig_size']))
        start = -0.2
        end = 0.25

        x_tick_range = np.arange(start, end, 0.05)
        y_tick_range = np.arange(start, end, 0.05)
        box_props = dict(boxstyle=self.visConfig['box_style'], facecolor=self.visConfig['face_color'])

        covars_dir = os.listdir(self.paths["covars_path"])
        rho_labels = []
        for label in self.seqConfig.keys():
            if not self.seqConfig[label]['run_model']:
                continue
            print("\t\t\t\t\tcovar corrs for:\ttest\t\tvs\t\t", label)
            if label.lower() == "svae":
                covars = np.load(self.paths["out_path"]+"covars_"+self.name+".npy")
            
            for cov in covars_dir:
                if label.lower() in cov.lower():
                    print(cov)
                    covars = np.load(self.paths["covars_path"] + cov)
                if "target" in cov.lower():
                    print(cov)
                    target_covars = np.load(self.paths["covars_path"] + cov)
                    
            
            if label == "Indep":
                target_masked = target_covars[:10000]
                covars_masked = covars[:10000]
            else:
                covars_masked = np.ma.masked_inside(covars, -0.02, 0.02).ravel()
                target_masked = np.ma.masked_inside(target_covars, -0.02, 0.02).ravel()

            pearson_r, pearson_p = pearsonr(target_covars.ravel(), covars.ravel())
            c = self.seqConfig[label]['color_set']
            marker_size = self.visConfig['marker_size'] - 4

            if label != "Target":
                marker_size = marker_size * 2.5
            # if label == "Val":
            #     marker_size = marker_size * 2.5
            # if label == "sVAE":
            #     marker_size = marker_size * 4
            # if label == "Indep":
            #     marker_size = marker_size * 5

            print(pearson_r, pearson_p)
            label_text = label + ", " + r"$\rho$ = " + str(round(pearson_r, self.visConfig['stats_sf']))    # orig with rho
            rho_labels.append(label_text)
            ax.plot(target_masked, covars_masked, 'o', markersize=marker_size, color=c, label=label_text, zorder=self.seqConfig[label]['z_order'], alpha=0.5)
        
        
        ax.set_rasterization_zorder(0)
        title_text = "Covariance Correlations Scatterplot\n"
        box_y = 0.10
        box_x = -0.10
        xlabel = "Target Covariances"
        pylab.xlabel(xlabel, fontsize=self.visConfig['label_size'])
        pylab.ylabel("Other Covariances", fontsize=self.visConfig['label_size'])
        pylab.xticks(x_tick_range, rotation=self.visConfig['tick_rotation'], fontsize=self.visConfig['tick_size'])
        pylab.yticks(y_tick_range, fontsize=self.visConfig['tick_size'])
        pylab.tick_params(direction='in', axis='both', which='major', labelsize=self.visConfig['tick_size'], length=self.visConfig['tick_length'], width=self.visConfig['tick_width'])
        lim_start = -0.2
        lim_end = 0.2
        pylab.xlim((lim_start, lim_end))
        pylab.ylim((lim_start, lim_end))
        pylab.tight_layout()
        leg_fontsize = self.visConfig['tick_size'] - 2
        pylab.legend(fontsize=leg_fontsize-2, loc="upper left", title_fontsize=leg_fontsize-2, labels=rho_labels )
        save_name = self.paths["vis_path"] + "covars_" + self.name + ".pdf"
        if self.visConfig["show_hp"]:
            fig.suptitle(self.name, fontsize = 8)
        pylab.savefig(save_name, dpi=self.visConfig['dpi'], format='pdf', bbox_inches='tight')
        pylab.close()
        print("\t\tcompleted: plotting covars")