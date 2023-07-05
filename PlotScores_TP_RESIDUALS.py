#!/usr/bin/env python

import os
import sys
from pathlib import Path
import pickle
from pprint import pprint

from collections import defaultdict
from numpy import median, mean, std, arange

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# =================================================
# =================================================
# =================================================

METHOD = "median"  # "mean"

RANDOM_NUMBERS = [ "36", "50", "142"]  # , "234", "777", "987"]
DATASETS = [("INSTANCE", "INSTANCE"), ("INSTANCE", "ETHZ")]
SIZES = ["NANO3", "NANO2", "NANO1", "NANO", "MICRO", "TINY", "SMALL", "MEDIUM", "LARGE"]
THRVAL = ["02", "05"]

BASEFOLDER_START = "./FinalRetrain_DKPN_PN_PAPER___Rnd__"
BASEFOLDER_END = "__TESTING-ULTIMATE_noDetrend"

# =================================================
# =================================================
# =================================================


def create_residuals_plot_compare(resP_dkpn, resS_dkpn, resP_pn, resS_pn, size_name, axs,
                                  binwidth=0.03):

    bin_edges = arange(-11.05, 11.05 + binwidth, binwidth)

    for (ax, data, title) in zip(
                    axs, ((resP_dkpn, resP_pn), (resS_dkpn, resS_pn)), (size_name+'_P', size_name+'_S')):

        ax.hist(data[0], bins=bin_edges, color="orange", edgecolor=None,
                label="DKPN: count=%d\n            mean=%.2f\n             std=%.2f" % (len(data[0]), mean(data[0]), std(data[0])))
        ax.hist(data[1], bins=bin_edges, facecolor=(.0, .0, .0, .0), edgecolor='blue',
                label="PN: count=%d\n    mean=%.2f\n      std=%.2f" % (len(data[1]), mean(data[1]), std(data[1])))

        # Set labels and title
        ax.set_xlabel('residuals (s)')
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylabel('count')
        ax.set_title(title)
        ax.legend()


if __name__ == "__main__":

    for (_ds1, _ds2) in DATASETS:  # InDomain CrossDomain

        for _thr in THRVAL:

            fig, axs = plt.subplots(len(SIZES), 2, figsize=(12, 3*len(SIZES)))
            axs = axs.flatten()
            # store_file=("%s_%s_%s_%s_scores.pdf" % , METHOD.lower())))
            print("%s Training - %s Testing - THR: %.1f" % (_ds1, _ds2, float(_thr)*0.1))
            plt.suptitle("%s Training - %s Testing - THR: %.1f" % (_ds1, _ds2, float(_thr)*0.1),
                         fontweight='bold', fontsize=18, fontname="Lato")  #fontfamily='sans-serif')

            xx = 0
            for _sz in SIZES:

                print("... %s" % _sz.upper())

                list_DKPN_P, list_PN_P = [], []
                list_DKPN_S, list_PN_S = [], []

                for _rndm in RANDOM_NUMBERS:
                    _work_path = Path(BASEFOLDER_START+_rndm+BASEFOLDER_END) / ("Results_"+_ds1+"_"+_ds2+"_"+_sz+"_"+_thr)

                    # === P

                    pickle_DKPN = str(_work_path / "DKPN_TP_P_residuals.pickle")
                    with open(pickle_DKPN, 'rb') as file:
                        loaded_data = pickle.load(file)
                    list_DKPN_P.extend(loaded_data)

                    pickle_PN = str(_work_path / "PN_TP_P_residuals.pickle")
                    with open(pickle_PN, 'rb') as file:
                        loaded_data = pickle.load(file)
                    list_PN_P.extend(loaded_data)

                    # === S

                    pickle_DKPN = str(_work_path / "DKPN_TP_S_residuals.pickle")
                    with open(pickle_DKPN, 'rb') as file:
                        loaded_data = pickle.load(file)
                    list_DKPN_S.extend(loaded_data)

                    pickle_PN = str(_work_path / "PN_TP_S_residuals.pickle")
                    with open(pickle_PN, 'rb') as file:
                        loaded_data = pickle.load(file)
                    list_PN_S.extend(loaded_data)

                # --> End RANDOM -->  populate figure
                _ = create_residuals_plot_compare(list_DKPN_P, list_DKPN_S, list_PN_P, list_PN_S, _sz.upper(),
                                                  [axs[xx], axs[xx+1]], binwidth=0.01)
                xx += 2

            # --> End SIZES --> CLOSE FIGURE
            # ===============================================================

            plt.tight_layout()  # Adjust the spacing between subplots
            fig.savefig("%s_%s_%s_TP_residuals.pdf" % (_ds1, _ds2, _thr))

        # --> End THR

    # --> End DATASETs

print("DONE")
