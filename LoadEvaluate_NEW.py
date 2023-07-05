#!/usr/bin/env python

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import obspy

import torch
import seisbench as sb
import seisbench.models as sbm

import dkpn.core as dkcore
import dkpn.train as dktrain

import dkpn.eval_utils as EV


print(" SB version:  %s" % sb.__version__)
print("OBS version:  %s" % obspy.__version__)
print("")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=(
                                "Script for comparing DKPN and PhaseNet models. "
                                "It needs to have the 'dkpn' folder in the working path. "
                                "The residuals plot "
                                "Requires Python >= 3.9"))

parser.add_argument('-d', '--dataset_name', type=str, default='ETHZ', help='Dataset name for TEST')
parser.add_argument('-s', '--dataset_size', type=str, default='Nano', help='Dataset size')
parser.add_argument('-r', '--random_seed', type=int, default=42, help='Random seed')
parser.add_argument('-o', '--store_folder', type=str, default='trained_results', help='Comparison Results folder')
#
parser.add_argument('-k', '--dkpn_model_name', type=str, required=True, help='DKPN model path')
parser.add_argument('-p', '--pn_model_name', type=str, required=True, help='PN model path')
parser.add_argument('-x', '--pickthreshold_p', type=float, default=0.2, help='Pick threshold P')
parser.add_argument('-y', '--pickthreshold_s', type=float, default=0.2, help='Pick threshold S')
parser.add_argument('-a', '--truepositive_p', type=int, default=10, help='Delta for declare True Positive P (samples)')
parser.add_argument('-b', '--truepositive_s', type=int, default=20, help='Delta for declare True Positive S (samples)')
parser.add_argument('-n', '--test_samples', type=int, default=5000, help='Number of test samples')
parser.add_argument('-f', '--nplots', type=int, default=10, help='Number of examples plots')
#
args = parser.parse_args()

# Your main function here
print(f"DKPN_MODEL_NAME: {args.dkpn_model_name}")
print(f"PN_MODEL_NAME: {args.pn_model_name}")
print("")
print(f"DATASET_NAME: {args.dataset_name}")
print(f"DATASET_SIZE: {args.dataset_size}")
print(f"RANDOM_SEED: {args.random_seed}")
print(f"STORE_FOLDER: {args.store_folder}")
print("")
print(f"PICK_THR_P: {args.pickthreshold_p}")
print(f"PICK_THR_S: {args.pickthreshold_s}")
print(f"DELTA_TP_P: {args.truepositive_p}")
print(f"DELTA_TP_S: {args.truepositive_s}")
print(f"NPLOTS: {args.nplots}")
print(f"NSAMPLES: {args.test_samples}")


# DKPN_MODEL_PATH = Path(args.dkpn_model_name+"/"+args.dkpn_model_name+".pt")
# PN_MODEL_PATH = Path(args.pn_model_name+"/"+args.pn_model_name+".pt")

DKPN_MODEL_PATH = [xx for xx in Path(args.dkpn_model_name).glob("*.pt")][0]
PN_MODEL_PATH = [xx for xx in Path(args.pn_model_name).glob("*.pt")][0]


STORE_DIR_RESULTS = Path(args.store_folder)
if not STORE_DIR_RESULTS.is_dir():
    STORE_DIR_RESULTS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# SELECT DATASET and SIZE

dataset_train = dktrain.select_database_and_size(
                            args.dataset_name, args.dataset_size,
                            RANDOM_SEED=args.random_seed)

train, dev, test = dataset_train.train_dev_test()

print("TRAIN samples %s:  %d" % (args.dataset_name, len(train)))
print("  DEV samples %s:  %d" % (args.dataset_name, len(dev)))
print(" TEST samples %s:  %d" % (args.dataset_name, len(test)))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

print("Loading DKPN ... %s" % Path(args.dkpn_model_name).name)
mydkpn = dkcore.DKPN()
mydkpn.load_state_dict(torch.load(str(DKPN_MODEL_PATH), map_location=torch.device('cpu')))
mydkpn.eval();
mydkpn.cuda();


print("Loading PN ... %s" % Path(args.pn_model_name).name)
mypn = sbm.PhaseNet()
mypn.load_state_dict(torch.load(str(PN_MODEL_PATH), map_location=torch.device('cpu')))
mypn.eval();
mypn.cuda();

# =================================================================
# =================================================================


# -------------------------------------------------------------------------------
# 
# # EVALUATING MODEL
# 
# Checking everything is OK and doing statistics using _5000 random samples_ extracted from the `test_generator`. But first, we need to **close** the MODEL before any prediction!
# We need to define the TruePositive, FalsePositive, FalseNegative:
# 
# - **TP**: if a pick of the same label falls inside a 0.2 seconds 
# - **FP**: if model declare a pick that doesn't have a match 
# - **FN**: if there's a label but unseen by the model
# 
# The functions and indexes are contained in `dkpn.eval_utils.py`.
# For consistency, we 
#
# In[11]:


# ========================  Getting / Instantiate class 
TRAIN_CLASS_DKPN = dktrain.TrainHelp_DomainKnowledgePhaseNet(
                mydkpn,  # It will contains the default args for StreamCF calculations!!!
                train,
                dev,
                test,

                augmentations_par={
                    "amp_norm_type": "std",
                    "window_strategy": "move",  # "pad"
                    "final_windowlength": 3001,
                    "sigma": 10,
                    "phase_dict": {
                        "trace_p_arrival_sample": "P",
                        "trace_pP_arrival_sample": "P",
                        "trace_P_arrival_sample": "P",
                        "trace_P1_arrival_sample": "P",
                        "trace_Pg_arrival_sample": "P",
                        "trace_Pn_arrival_sample": "P",
                        "trace_PmP_arrival_sample": "P",
                        "trace_pwP_arrival_sample": "P",
                        "trace_pwPm_arrival_sample": "P",
                        "trace_s_arrival_sample": "S",
                        "trace_S_arrival_sample": "S",
                        "trace_S1_arrival_sample": "S",
                        "trace_Sg_arrival_sample": "S",
                        "trace_SmS_arrival_sample": "S",
                        "trace_Sn_arrival_sample": "S"
                    },
                })

# ========================  AUGMENTATIONS DKPN
(train_generator_dkpn, dev_generator_dkpn, test_generator_dkpn) = TRAIN_CLASS_DKPN.get_generator()

# ========================  CREATE A LIST OF UNIQUE INDEX FROM RANDOM ... AVOID DUPLICATES
rng = np.random.default_rng(seed=args.random_seed)
rnidx = rng.choice(np.arange(args.test_samples), size=args.test_samples, replace=False)


# --------------------------------------------------------------

do_stats_on = [# (dev_generator_dkpn, "DEV_DKPN", "DEV_PN"),
               (test_generator_dkpn, "TEST_DKPN", "TEST_PN"),
                   ]

dkpn_p_pick_residuals, dkpn_s_pick_residuals = [], []
pn_p_pick_residuals, pn_s_pick_residuals = [], []


for (DKPN_gen, DKPN_gen_name, PN_gen_name) in do_stats_on:

    print("Working with:  %s + %s" % (DKPN_gen_name, PN_gen_name))
    DKPN_stats_dict_P, DKPN_stats_dict_S = EV.__reset_stats_dict__(), EV.__reset_stats_dict__()
    PN_stats_dict_P, PN_stats_dict_S = EV.__reset_stats_dict__(), EV.__reset_stats_dict__()

    figureidx = 0

    for xx in tqdm(range(args.test_samples)):

        DKPN_sample = DKPN_gen[rnidx[xx]]
        # Create equal window for PN (stored in Xorig of DKPN, but we must remove the fp_stab samples)
        PN_sample = {}
        PN_sample["X"] = DKPN_sample["Xorig"][:, mydkpn.default_args["fp_stabilization"]*mydkpn.sampling_rate:]
        PN_sample["y"] = DKPN_sample["y"]

        # --- DEMEAN  input stream matrix (3C) with the matrix standard deviation,
        #                to compensate the lack of this step in Augmentation
        PN_sample["X"] = PN_sample["X"] - np.mean(PN_sample["X"], axis=1, keepdims=True)

        # --- NORMALIZE  input stream matrix (3C) with the matrix standard deviation,
        #                to compensate the lack of this step in Augmentation
        PN_sample["X"] = PN_sample["X"] / (
            np.std(PN_sample["X"], axis=1, keepdims=True) + 1e-6
            )

        # ----------------- Do PREDICTIONS
        # print("... Doing Predictions!")
        with torch.no_grad():
            DKPN_pred = mydkpn(torch.tensor(DKPN_sample["X"], device=mydkpn.device).unsqueeze(0))  # Add a fake batch dimension
            DKPN_pred = DKPN_pred[0].cpu().numpy()

        with torch.no_grad():
            PN_pred = mypn(torch.tensor(PN_sample["X"], device=mydkpn.device).unsqueeze(0))  # Add a fake batch dimension
            PN_pred = PN_pred[0].cpu().numpy()

        # ------------------------------------------------------------
        # ----------------- Do STATISTICS DKPN

        # P
        (DKPN_P_picks_model, DKPN_P_widths_model) = EV.extract_picks(
                                                        DKPN_pred[0],
                                                        smooth=True,
                                                        thr=args.pickthreshold_p)
        (DKPN_P_picks_label, DKPN_P_widths_label) = EV.extract_picks(
                                                        DKPN_sample["y"][0],
                                                        smooth=True,
                                                        thr=args.pickthreshold_p)

        (DKPN_stats_dict_P, DKPN_residual_TP_P) = EV.compare_picks(
                                          DKPN_P_picks_model, 
                                          DKPN_P_picks_label, 
                                          DKPN_stats_dict_P,
                                          thr=args.truepositive_p)
        # S
        (DKPN_S_picks_model, DKPN_S_widths_model) = EV.extract_picks(
                                                        DKPN_pred[1],
                                                        smooth=True,
                                                        thr=args.pickthreshold_s)
        (DKPN_S_picks_label, DKPN_S_widths_label) = EV.extract_picks(
                                                        DKPN_sample["y"][1],
                                                        smooth=True,
                                                        thr=args.pickthreshold_s)

        (DKPN_stats_dict_S, DKPN_residual_TP_S) = EV.compare_picks(
                                          DKPN_S_picks_model, 
                                          DKPN_S_picks_label, 
                                          DKPN_stats_dict_S,
                                          thr=args.truepositive_s)

        dkpn_p_pick_residuals.extend(DKPN_residual_TP_P)
        dkpn_s_pick_residuals.extend(DKPN_residual_TP_S)

        # ------------------------------------------------------------
        # ----------------- Do STATISTICS PN

        # P
        (PN_P_picks_model, PN_P_widths_model) = EV.extract_picks(
                                                        PN_pred[0],
                                                        smooth=True,
                                                        thr=args.pickthreshold_p)
        (PN_P_picks_label, PN_P_widths_label) = EV.extract_picks(
                                                        PN_sample["y"][0],
                                                        smooth=True,
                                                        thr=args.pickthreshold_p)

        (PN_stats_dict_P, PN_residual_TP_P) = EV.compare_picks(
                                        PN_P_picks_model, 
                                        PN_P_picks_label, 
                                        PN_stats_dict_P,
                                        thr=args.truepositive_p)
        # S
        (PN_S_picks_model, PN_S_widths_model) = EV.extract_picks(
                                                        PN_pred[1],
                                                        smooth=True,
                                                        thr=args.pickthreshold_s)
        (PN_S_picks_label, PN_S_widths_label) = EV.extract_picks(
                                                        PN_sample["y"][1],
                                                        smooth=True,
                                                        thr=args.pickthreshold_s)

        (PN_stats_dict_S, PN_residual_TP_S) = EV.compare_picks(
                                        PN_S_picks_model, 
                                        PN_S_picks_label,
                                        PN_stats_dict_S,
                                        thr=args.truepositive_s)

        pn_p_pick_residuals.extend(PN_residual_TP_P)
        pn_s_pick_residuals.extend(PN_residual_TP_S)

        # ------------------------------------------------------------
        # ----------------- PLOTS

        if (figureidx+1) <= args.nplots:
            fig = EV.create_AL_plots(
                    PN_sample["X"],
                    PN_sample["y"],
                    DKPN_sample["X"],
                    PN_pred,
                    DKPN_pred,
                    PN_P_picks_label,    # The groundtruth IDX
                    PN_S_picks_label,    # The groundtruth IDX
                    PN_P_picks_model,    # The PN model picks IDX
                    PN_S_picks_model,    # The PN model picks IDX
                    DKPN_P_picks_model,  # The DKPN model picks IDX
                    DKPN_S_picks_model,  # The DKPN model picks IDX
                    save_path=str(
                        STORE_DIR_RESULTS / ("Prediction_Example_%s_%s_%d.pdf" % (
                                           DKPN_gen_name, PN_gen_name, figureidx))
                                ))
        #
        figureidx += 1
    
    # Convert list of residuals, into numpy array of seconds
    dkpn_p_pick_residuals = np.array(dkpn_p_pick_residuals)*0.01
    dkpn_s_pick_residuals = np.array(dkpn_s_pick_residuals)*0.01
    pn_p_pick_residuals = np.array(pn_p_pick_residuals)*0.01
    pn_s_pick_residuals = np.array(pn_s_pick_residuals)*0.01

    # ------------------------------------------
    # ------- FINAL STATISTICS ON DKPN
    (DKPN_P_f1, DKPN_P_precision, DKPN_P_recall) = EV.calculate_scores(DKPN_stats_dict_P)
    (DKPN_S_f1, DKPN_S_precision, DKPN_S_recall) = EV.calculate_scores(DKPN_stats_dict_S)

    with open(str(STORE_DIR_RESULTS / ("SCORES_%s.txt" % DKPN_gen_name)), "w") as OUT:
        OUT.write(("samples:  %d"+os.linesep*2) % args.test_samples)
        #
        for vv, kk in DKPN_stats_dict_P.items():
            vv = "P_"+vv
            OUT.write(("%7s:  %7d"+os.linesep) % (vv, kk))
        #
        OUT.write(os.linesep)
        OUT.write(("P_f1:         %4.2f"+os.linesep) % DKPN_P_f1)
        OUT.write(("P_precision:  %4.2f"+os.linesep) % DKPN_P_precision)
        OUT.write(("P_recall:     %4.2f"+os.linesep*2) % DKPN_P_recall)
        #
        for vv, kk in DKPN_stats_dict_S.items():
            vv = "S_"+vv
            OUT.write(("%7s:  %7d"+os.linesep) % (vv, kk))
        #
        OUT.write(os.linesep)
        OUT.write(("S_f1:         %4.2f"+os.linesep) % DKPN_S_f1)
        OUT.write(("S_precision:  %4.2f"+os.linesep) % DKPN_S_precision)
        OUT.write(("S_recall:     %4.2f"+os.linesep*2) % DKPN_S_recall)

    # CREATE dictionary to disk
    res_dict = {}
    res_dict['samples'] = args.test_samples
    #
    res_dict.update({"P_"+kk: vv for kk, vv in DKPN_stats_dict_P.items()})
    res_dict["P_f1"] = DKPN_P_f1
    res_dict["P_precision"] = DKPN_P_precision
    res_dict["P_recall"] = DKPN_P_recall
    #
    res_dict.update({"S_"+kk: vv for kk, vv in DKPN_stats_dict_S.items()})
    res_dict["S_f1"] = DKPN_S_f1
    res_dict["S_precision"] = DKPN_S_precision
    res_dict["S_recall"] = DKPN_S_recall

    # SAVE dictionary to disk
    with open(str(STORE_DIR_RESULTS / 'results_DKPN.pickle'), 'wb') as file:
        pickle.dump(res_dict, file)

    # ------------------------------------------
    # ------- FINAL STATISTICS ON PN
    (PN_P_f1, PN_P_precision, PN_P_recall) = EV.calculate_scores(PN_stats_dict_P)
    (PN_S_f1, PN_S_precision, PN_S_recall) = EV.calculate_scores(PN_stats_dict_S)

    with open(str(STORE_DIR_RESULTS / ("SCORES_%s.txt" % PN_gen_name)), "w") as OUT:
        OUT.write(("samples:  %d"+os.linesep*2) % args.test_samples)
        for vv, kk in PN_stats_dict_P.items():
            vv = "P_"+vv
            OUT.write(("%7s:  %7d"+os.linesep) % (vv, kk))
        OUT.write(os.linesep)
        OUT.write(("P_f1:         %4.2f"+os.linesep) % PN_P_f1)
        OUT.write(("P_precision:  %4.2f"+os.linesep) % PN_P_precision)
        OUT.write(("P_recall:     %4.2f"+os.linesep*2) % PN_P_recall)
        #
        for vv, kk in PN_stats_dict_S.items():
            vv = "S_"+vv
            OUT.write(("%7s:  %7d"+os.linesep) % (vv, kk))
        OUT.write(os.linesep)
        OUT.write(("S_f1:         %4.2f"+os.linesep) % PN_S_f1)
        OUT.write(("S_precision:  %4.2f"+os.linesep) % PN_S_precision)
        OUT.write(("S_recall:     %4.2f"+os.linesep*2) % PN_S_recall)

    # CREATE dictionary to disk
    res_dict = {}
    res_dict['samples'] = args.test_samples
    #
    res_dict.update({"P_"+kk: vv for kk, vv in PN_stats_dict_P.items()})
    res_dict["P_f1"] = PN_P_f1
    res_dict["P_precision"] = PN_P_precision
    res_dict["P_recall"] = PN_P_recall
    #
    res_dict.update({"S_"+kk: vv for kk, vv in PN_stats_dict_S.items()})
    res_dict["S_f1"] = PN_S_f1
    res_dict["S_precision"] = PN_S_precision
    res_dict["S_recall"] = PN_S_recall

    # SAVE dictionary to disk
    with open(str(STORE_DIR_RESULTS / 'results_PN.pickle'), 'wb') as file:
        pickle.dump(res_dict, file)

    # SAVE RESIDUALS - DKPN
    with open(str(STORE_DIR_RESULTS / 'DKPN_TP_P_residuals.pickle'), 'wb') as file:
        pickle.dump(dkpn_p_pick_residuals, file)

    with open(str(STORE_DIR_RESULTS / 'DKPN_TP_S_residuals.pickle'), 'wb') as file:
        pickle.dump(dkpn_s_pick_residuals, file)

    # SAVE RESIDUALS - PN
    with open(str(STORE_DIR_RESULTS / 'PN_TP_P_residuals.pickle'), 'wb') as file:
        pickle.dump(pn_p_pick_residuals, file)

    with open(str(STORE_DIR_RESULTS / 'PN_TP_S_residuals.pickle'), 'wb') as file:
        pickle.dump(pn_s_pick_residuals, file)

    # =============================================================================
    fig = EV.create_residuals_plot_compare(dkpn_p_pick_residuals, dkpn_s_pick_residuals, 
                                           pn_p_pick_residuals, pn_s_pick_residuals, 
                                           binwidth=0.025,
                                           save_path=str(STORE_DIR_RESULTS / "Residuals_P_S_comparison_DKPN_PN.pdf"))

# Store PARAMETER
with open(str(STORE_DIR_RESULTS / "CALL_ARGS.py"), "w") as OUT:
    OUT.write("ARGS=%s" % args)
