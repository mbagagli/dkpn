from tqdm import tqdm
import os
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader

from pathlib import Path

import seisbench as sb
import seisbench.data as sbd
import seisbench.generate as sbg

from dkpn.core import PreProc

 
# ==================================================================
# ==================================================================
# ==================================================================

def __add_split_column__(indata,
                         TRAIN_PERC=0.60,
                         DEV_PERC=0.30,
                         TEST_PERC=0.10,
                         RANDOM_SEED=42,
                         verbose=False):
    print("Adding SPLIT column for SeisBench APIs")
    indata.metadata["split"] = "dev"
    _idxcol = indata.metadata.columns.get_loc("split")

    # =======================================================  Divide  the TRAIN/TESTSPLIT
    LEN_TRAIN = int(len(indata.metadata)*TRAIN_PERC)
    indata.metadata.iloc[:LEN_TRAIN, _idxcol] = "train"
    if verbose:
        print("Length of TRAINING DATASET:  %d" % LEN_TRAIN)

    LEN_TEST = int(len(indata.metadata)*TEST_PERC)
    indata.metadata.iloc[(LEN_TRAIN+1):(LEN_TRAIN+LEN_TEST+1), _idxcol] = "test"
    if verbose:
        print("Length of TEST DATASET:  %d" % LEN_TEST)

    LEN_DEV = len(indata.metadata) - LEN_TRAIN - LEN_TEST
    indata.metadata.iloc[(LEN_TRAIN+LEN_TEST):, _idxcol] = "dev"
    if verbose:
        print("Length of DEV DATASET:  %d" % LEN_DEV)
    return indata


def __filter_sb_dataset__(indata, filter_for="INSTANCE"):

    if filter_for.lower() == "instance":
        indata.filter(indata.metadata["path_ep_distance_km"] <= 100.0, inplace=True)
        indata.filter(indata.metadata["station_channels"] == "HH", inplace=True)

        # P and S
        indata.filter(np.logical_not(np.logical_or(   # not (no P or no S)
            np.isnan(indata.metadata["trace_P_arrival_sample"]),
            np.isnan(indata.metadata["trace_S_arrival_sample"]))
        ), inplace=True)

        # P-S < 3000 (both P and S inside the 30 second window)
        indata.filter((indata.metadata["trace_S_arrival_sample"] -
                       indata.metadata["trace_P_arrival_sample"] < 3000),
                      inplace=True)

    elif filter_for.lower() == "ethz":
        indata.filter(indata.metadata["trace_completeness"] >= 0.99, inplace=True)

    elif filter_for.lower() == "pnw":
        indata.filter(indata.metadata["source_type"] == "earthquake", inplace=True)
        indata.filter(indata.metadata["trace_has_offset"] == 0, inplace=True)
        # Avoid missing channel, to help S-detection
        indata.filter(indata.metadata["trace_missing_channel"] < 1, inplace=True)

    elif filter_for.lower() == "aquila":
        indata.filter(indata.metadata["source_type"] == "earthquake", inplace=True)
        indata.filter(indata.metadata["path_ep_distance_km"] <= 100.0, inplace=True)
        # indata.filter(
        #     (indata.metadata["trace_p2_arrival_seconds"] -
        #      indata.metadata["trace_p1_arrival_seconds"]) <= 15, inplace=True),
        indata.filter(indata.metadata["PICK_COUNT"] >= 4, inplace=True)
        indata.filter(indata.metadata["trace_p_weight"] <= 3, inplace=True)

    else:
        raise ValueError("Invalid Filtering key!")

    return indata


def __select_database_and_size_PNW__(dataset_size, filtering=False,
                                     use_biggest_anyway=True,
                                     RANDOM_SEED=42,
                                     train_perc_size=0.60,
                                     dev_perc_size=0.30,
                                     test_perc_size=0.10):

    dataset_train = sbd.WaveformDataset("/scratch/seisbench/datasets/pnw",
                                        sampling_rate=100, cache="trace")
    dataset_train = __add_split_column__(dataset_train,
                                         TRAIN_PERC=train_perc_size,
                                         DEV_PERC=dev_perc_size,
                                         TEST_PERC=test_perc_size,
                                         RANDOM_SEED=RANDOM_SEED)

    # FILTER
    if filtering:
        dataset_train = __filter_sb_dataset__(dataset_train,
                                              filter_for="PNW")
        dataset_train.metadata.reset_index(inplace=True)

    (_train, _dev, _test) = dataset_train.train_dev_test()
    return (_train, _dev, _test)


def __select_database_and_size_AQUILA__(dataset_size, filtering=False,
                                        use_biggest_anyway=True,
                                        RANDOM_SEED=42,
                                        train_perc_size=0.60,
                                        dev_perc_size=0.10,
                                        test_perc_size=0.30):

    dataset_train = sbd.WaveformDataset("/scratch/seisbench/datasets/aq2009counts",
                                        sampling_rate=100.0, cache="trace")
    dataset_train = __add_split_column__(dataset_train,
                                         TRAIN_PERC=train_perc_size,
                                         DEV_PERC=dev_perc_size,
                                         TEST_PERC=test_perc_size,
                                         RANDOM_SEED=RANDOM_SEED)

    # FILTER
    if filtering:
        dataset_train = __filter_sb_dataset__(dataset_train,
                                              filter_for="AQUILA")
        dataset_train.metadata.reset_index(inplace=True)

    (_train, _dev, _test) = dataset_train.train_dev_test()
    return (_train, _dev, _test)


def __select_database_and_size_ETHZ__(dataset_size, filtering=False,
                                      use_biggest_anyway=True,
                                      RANDOM_SEED=42):

    dataset_train = sbd.ETHZ(sampling_rate=100, cache="trace")

    # FILTER
    if filtering:
        dataset_train = __filter_sb_dataset__(dataset_train,
                                              filter_for="ETHZ")
        dataset_train.metadata.reset_index(inplace=True)

    if dataset_size.lower() == "nano3":  # NANO3 -> 803 80*2
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.021875, 0.0043750, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano2":  # NANO2 -> 1607 160*2
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.04375, 0.008750, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano1":  # NANO1 -> 3215 321*2
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.0875, 0.01750, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano":  # NANO -> 6430 643*2
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.175, 0.0350, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "micro":  # MICRO -> 12860 1286*2
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.35, 0.070, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "tiny":  # TINY -> 25720 2572*2
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.7, 0.14, 0.0),
                                            random_seed=RANDOM_SEED)

    else:
        if use_biggest_anyway:
            # Use the biggest possible --> in our case TINY
            dataset_train._set_splits_random_sampling(
                                                ratios=(0.7, 0.14, 0.0),
                                                random_seed=RANDOM_SEED)
        else:
            raise ValueError("Not a valid DATASET SIZE for ETHZ!")

    (_train, _dev, _test) = dataset_train.train_dev_test()
    return (_train, _dev, _test)


def __select_database_and_size_INSTANCE__(dataset_size, filtering=False,
                                          RANDOM_SEED=42):

    dataset_train = sbd.InstanceCounts(sampling_rate=100, cache="trace")

    # FILTER
    if filtering:
        dataset_train = __filter_sb_dataset__(dataset_train,
                                              filter_for="INSTANCE")
        dataset_train.metadata.reset_index(inplace=True)

    # SIZE SPLIT
    if dataset_size.lower() == "nano3":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.0025, 0.00050, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano2":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.005, 0.0010, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano1":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.01, 0.002, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.02, 0.004, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "micro":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.04, 0.008, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "tiny":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.08, 0.016, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "small":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.2, 0.04, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "medium":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.5, 0.1, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "large":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.8, 0.16, 0.0),
                                            random_seed=RANDOM_SEED)

    else:
        raise ValueError("Not a valid DATASET SIZE for INSTANCE!")

    (_train, _dev, _test) = dataset_train.train_dev_test()
    return (_train, _dev, _test)


def __select_database_and_size_SCEDC__(dataset_size, filtering=False,
                                       RANDOM_SEED=42):

    dataset_train = sbd.SCEDC(sampling_rate=100, cache="trace")

    # FILTER
    if filtering:
        dataset_train = __filter_sb_dataset__(dataset_train)
        dataset_train.metadata.reset_index(inplace=True)

    # SIZE SPLIT
    if dataset_size.lower() == "nano3":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.0025, 0.00025, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano2":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.005, 0.0005, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano1":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.01, 0.001, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "nano":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.02, 0.002, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "micro":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.04, 0.004, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "tiny":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.08, 0.008, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "small":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.2, 0.02, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "medium":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.5, 0.05, 0.0),
                                            random_seed=RANDOM_SEED)

    elif dataset_size.lower() == "large":
        dataset_train._set_splits_random_sampling(
                                            ratios=(0.8, 0.1, 0.0),
                                            random_seed=RANDOM_SEED)

    else:
        raise ValueError("Not a valid DATASET SIZE for INSTANCE!")

    (_train, _dev, _test) = dataset_train.train_dev_test()
    return (_train, _dev, _test)


def select_database_and_size(dataset_name, dataset_size, RANDOM_SEED=42):
    """ Big Switch for selection of dataset and sample numbers """

    print("Selecting DATASET from:  %s" % sb.cache_root)
    print("Selecting DATASET NAME:  %s" % dataset_name.upper())
    print("Selecting DATASET SIZE:  %s" % dataset_size.upper())

    # ===========> DATASET
    if dataset_name.upper() == "ETHZ":
        (_train, _dev, _test) = __select_database_and_size_ETHZ__(
                                                    dataset_size.lower(),
                                                    filtering=True,
                                                    RANDOM_SEED=RANDOM_SEED)

    elif dataset_name.upper() == "INSTANCE":
        (_train, _dev, _test) = __select_database_and_size_INSTANCE__(
                                                    dataset_size.lower(),
                                                    filtering=True,
                                                    RANDOM_SEED=RANDOM_SEED)

    elif dataset_name.upper() == "SCEDC":
        (_train, _dev, _test) = __select_database_and_size_INSTANCE__(
                                                    dataset_size.lower(),
                                                    filtering=True,
                                                    RANDOM_SEED=RANDOM_SEED)
    elif dataset_name.upper() == "PNW":
        (_train, _dev, _test) = __select_database_and_size_PNW__(
                                                    dataset_size.lower(),
                                                    filtering=True,
                                                    RANDOM_SEED=RANDOM_SEED)
    elif dataset_name.upper() == "AQUILA":
        (_train, _dev, _test) = __select_database_and_size_AQUILA__(
                                                    dataset_size.lower(),
                                                    filtering=True,
                                                    RANDOM_SEED=RANDOM_SEED)

    else:
        raise ValueError("Not a valid DATASET NAME!")

    return (_train, _dev, _test)


def early_stop_criteria(trainl, trainl_batch,
                        devl, devl_batch,
                        patience, delta):
    """ Return a bool to stop or not the training """

    # Calculate mean improvement in the last '2*patience' epochs
    mean_loss_before = np.mean(devl[-2*patience:-patience-1])
    mean_loss_current = np.mean(devl[-patience:])
    improvement = mean_loss_before - mean_loss_current
    print("Early stopping: mean_DEV_LOSS_before %7f "
          "mean_DEV_LOSS_current %7f improvement %7f" % (mean_loss_before,
                                                         mean_loss_current,
                                                         improvement))

    # 1. If improvement is less than 'delta', stop the training
    if improvement < delta:
        print("@@@ Early stopping triggered! Flat DEV loss")
        return (True, "flat", 1)

    # 2. Check if the DEV LOSS is always higher than the highest TRAIN
    #    batch loss
    A = np.array([np.max(cc) for cc in trainl_batch[-patience:]])
    B = np.array(devl[-patience:])
    if np.all((B-A) > 0):
        print("@@@ Early stopping triggered! DEV loss always on top of TRAIN")
        return (True, "invert", patience)
    #
    return (False, "null", 0)

# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================


class TrainHelp_DomainKnowledgePhaseNet(object):

    def __init__(
            self,
            dkpninstance,  # It will contains the default args for StreamCF calculations!!!
            train_sb_data,
            dev_sb_data,
            test_sb_data,
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
                },
            batch_size=128,
            num_workers=24,
            random_seed=42):

        """ Modulus to prepare and process the data """
        self.augmentations_par = augmentations_par
        self.train_generator = sbg.GenericGenerator(train_sb_data)
        self.dev_generator = sbg.GenericGenerator(dev_sb_data)
        self.test_generator = sbg.GenericGenerator(test_sb_data)
        self.random_seed = random_seed
        self.train_loader, self.dev_loader, self.test_loader = None, None, None

        # ----------  0. Define query windows
        self.trainmod = dkpninstance
        self.model_epochs_list = []
        self.augmentations_par["fp_stabilization"] = int(
                            self.trainmod.default_args["fp_stabilization"]*100.0)

        # ---------  1. Define augmentations
        self.augmentations = self.__define_augmentations__(**self.augmentations_par)

        # ---------  2. Load AUGMENTATIONS
        self.train_generator.add_augmentations(self.augmentations)
        self.dev_generator.add_augmentations(self.augmentations)
        self.test_generator.add_augmentations(self.augmentations)

        # ---------  3. Create DATALOADER
        self.train_loader = DataLoader(self.train_generator, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       worker_init_fn=self.__worker_init_fn_seed__)
        self.dev_loader = DataLoader(self.dev_generator, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     worker_init_fn=self.__worker_init_fn_seed__)
        self.test_loader = DataLoader(self.test_generator, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      worker_init_fn=self.__worker_init_fn_seed__)

    def __worker_init_fn_seed__(self, wid):
        np.random.seed(self.random_seed)

    def __worker_init_fn_full_seed__(self, wid):
        """ Just know, no chanche to modify RanomWindow seed number inside """
        def seed_everything(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        #
        seed_everything(self.random_seed)

    def __define_augmentations__(self, **kwargs):
        """ Define which augmentations to use in the training class """
        samples_before = int(2 * kwargs["final_windowlength"] / 3) + kwargs["fp_stabilization"]
        windowlen = samples_before + int(2 * kwargs["final_windowlength"] / 3)
        rw_windowlen = kwargs["final_windowlength"] + kwargs["fp_stabilization"]
        rw_low = 0
        rw_high = windowlen

        augmentations_list = [
                sbg.WindowAroundSample(list(kwargs["phase_dict"].keys()),
                                       samples_before=windowlen/2,
                                       windowlen=windowlen,
                                       selection="first",
                                       strategy=kwargs["window_strategy"]),

                sbg.RandomWindow(windowlen=rw_windowlen,
                                 low=rw_low, high=rw_high,
                                 strategy=kwargs["window_strategy"]),

                sbg.Normalize(demean_axis=-1,
                              amp_norm_axis=-1,
                              amp_norm_type=kwargs["amp_norm_type"]),

                sbg.ProbabilisticLabeller(label_columns=kwargs["phase_dict"], 
                                          sigma=kwargs["sigma"], dim=0),

                sbg.ChangeDtype(np.float32, key="X"),
                sbg.ChangeDtype(np.float32, key="y"),
                sbg.Copy(key=("X", "Xorig")),

                # This Pre-Proc stage consists in:
                # - demean + std normalization of the 3C traces (like standard PhaseNet)
                # - calculations of the 5 CFs
                # - removing the first N samples + taking 3001 samples (final window)
                # - Std normalization of the 3 cfs + modulus
                PreProc(**self.trainmod.default_args),
            ]
        #
        return augmentations_list

    def extract_windows_cfs(self):
        outdict = {}

        for (gen, gen_name) in ((self.train_generator, "train"),
                                (self.dev_generator, "dev")):

            outdict[gen_name+"_X"], outdict[gen_name+"_Y"] = [], []

            for xx in tqdm(range(len(gen))):
                outdict[gen_name+"_X"].append(gen[xx]["X"])
                outdict[gen_name+"_Y"].append(gen[xx]["y"])
        #
        return outdict

    def __loss_fn__(self, y_pred, y_true, eps=1e-5):
        # vector cross entropy loss
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def __train_loop__(self, optimizer):
        size = len(self.train_loader.dataset)
        train_loss = 0
        train_loss_batches = []
        for batch_id, batch in enumerate(self.train_loader):
            # Compute prediction and loss
            pred = self.trainmod(batch["X"].to(
                                        self.trainmod.device))
            loss = self.__loss_fn__(pred, batch["y"].to(
                                        self.trainmod.device))
            train_loss_batches.append(loss.item())
            train_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                loss_val, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return (loss.item(), train_loss_batches)

    def __test_loop__(self):

        num_batches = len(self.dev_loader)
        test_loss = 0
        test_loss_batches = []

        self.trainmod.eval()  # 20230524

        with torch.no_grad():
            for batch in self.dev_loader:
                pred = self.trainmod(batch["X"].to(
                                self.trainmod.device))
                _tlb = self.__loss_fn__(pred, batch["y"].to(
                                self.trainmod.device)).item()
                #
                test_loss += _tlb
                test_loss_batches.append(_tlb)

        self.trainmod.train()  # 20230524

        test_loss /= num_batches
        print(f"Dev. avg loss: {test_loss:>8f}\n")
        return (test_loss, test_loss_batches)

    def train_me(self,
                 # Train related
                 epochs=15,
                 optimizer_type="adam",
                 learning_rate=1e-2):

        """ Daje """

        # Defining OPTIMIZER
        if optimizer_type.lower() in ("adam", "adm"):
            optim = torch.optim.Adam(self.trainmod.parameters(),
                                     lr=learning_rate)
        else:
            raise ValueError("At the moment only the 'Adam' optimizer "
                             "is supported!")

        # ------------------------ GO
        train_loss_epochs, train_loss_epochs_batches = [], []
        test_loss_epochs, test_loss_epochs_batches = [], []

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            (_train_loss, _train_loss_batches) = self.__train_loop__(
                                                            optimizer=optim)
            train_loss_epochs.append(_train_loss)
            train_loss_epochs_batches.append(_train_loss_batches)

            (_test_loss, _test_loss_batches) = self.__test_loop__()
            test_loss_epochs.append(_test_loss)
            test_loss_epochs_batches.append(_test_loss_batches)
        #
        self.__training_epochs__ = t+1
        return (train_loss_epochs, train_loss_epochs_batches,
                test_loss_epochs, test_loss_epochs_batches)

    def train_me_early_stop(
                self,
                epochs=15,
                optimizer_type="adam",
                learning_rate=1e-2,
                patience=5,
                delta=0.001):  # Threshold for minimum improvement

        """ Daje """

        # Defining OPTIMIZER
        if optimizer_type.lower() in ("adam", "adm"):
            optim = torch.optim.Adam(self.trainmod.parameters(),
                                     lr=learning_rate)
        else:
            raise ValueError("At the moment only the 'Adam' optimizer "
                             "is supported!")

        # ------------------------ GO
        train_loss_epochs, train_loss_epochs_batches = [], []
        test_loss_epochs, test_loss_epochs_batches = [], []

        losses = []  # track losses for the last 'patience' epochs

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            (_train_loss, _train_loss_batches) = self.__train_loop__(
                                                            optimizer=optim)
            train_loss_epochs.append(_train_loss)
            train_loss_epochs_batches.append(_train_loss_batches)

            (_test_loss, _test_loss_batches) = self.__test_loop__()
            test_loss_epochs.append(_test_loss)
            test_loss_epochs_batches.append(_test_loss_batches)

            losses.append(_test_loss)

            # Store model for HISTORY porpuses
            self.model_epochs_list.append(
                copy.deepcopy(
                    self.trainmod))

            # If we have more than '2*patience' epochs done,
            # we can start checking for early stopping

            if t >= 2 * patience:
                (_answer, _, _step_back) = early_stop_criteria(
                                train_loss_epochs, train_loss_epochs_batches,
                                test_loss_epochs, test_loss_epochs_batches,
                                patience, delta)

                if _answer:
                    # Early stop triggered..pack everything
                    self.__training_epochs__ = (t+1) - _step_back
                    self.trainmod = copy.deepcopy(
                                        self.model_epochs_list[-_step_back-1])
                    del self.model_epochs_list
                    return (train_loss_epochs[:-_step_back],
                            train_loss_epochs_batches[:-_step_back],
                            test_loss_epochs[:-_step_back],
                            test_loss_epochs_batches[:-_step_back])

                else:
                    # Free unnecessary memory
                    if len(self.model_epochs_list) > patience:
                        del self.model_epochs_list[:-patience-1]

        # If here, we reached the maximum epochs provided by the user,
        # Return everything in full
        print("@@@ Reached the FULL  %d  epochs!" % epochs)
        self.__training_epochs__ = t+1
        return (train_loss_epochs, train_loss_epochs_batches,
                test_loss_epochs, test_loss_epochs_batches)

    def store_weigths(self, dir_path, model_name, jsonstring, version="1"):
        """ Store the finals """

        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)

        def _create_json_(docs, pathfile):
            def_dct = self.trainmod.get_defaults()
            with open(str(pathfile), "w") as OUT:

                OUT.write("{"+os.linesep)
                OUT.write(("    \"docstring\": \"%s\","+os.linesep) % docs)
                OUT.write("    \"model_args\": {"+os.linesep)
                OUT.write("        \"component_order\": \"ZNE\","+os.linesep)
                OUT.write("        \"phases\": \"PSN\""+os.linesep)
                OUT.write("    },"+os.linesep)
                OUT.write("    \"seisbench_requirement\": \"0.3.0\","+os.linesep)
                OUT.write(("    \"version\": \"%s\","+os.linesep) % version)
                OUT.write("    \"default_args\": {"+os.linesep)
                #
                for kk, vv in def_dct.items():
                    if isinstance(vv, bool):
                        OUT.write(("        \"%s\": %s,"+os.linesep) % (kk, str(vv).lower()))
                    elif isinstance(vv, str):
                        OUT.write(("        \"%s\": %s,"+os.linesep) % (kk, repr(vv).replace("'", '"')))
                    else:
                        OUT.write(("        \"%s\": %r,"+os.linesep) % (kk, vv))
                OUT.write("        \"blinding\": ["+os.linesep)
                OUT.write("            250,"+os.linesep)
                OUT.write("            250"+os.linesep)
                OUT.write("        ]"+os.linesep)
                #
                OUT.write("    }"+os.linesep)
                OUT.write("}"+os.linesep)
        #
        if not dir_path.is_dir:
            dir_path.mkdir()

        _create_json_(jsonstring, dir_path / (str(model_name) + ".json"))
        torch.save(self.trainmod.state_dict(),
                   str(dir_path / (str(model_name) + ".pt"))
                   )

    def get_model(self):
        return self.trainmod

    def get_generator(self):
        return (self.train_generator, self.dev_generator, self.test_generator)

    def get_loader(self):
        return (self.train_loader, self.dev_loader, self.test_loader)

# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================    
# ==================================================================
# ==================================================================
# ==================================================================


class TrainHelp_PhaseNet(object):

    def __init__(
            self,
            pninstance,  # It will contains the default args for StreamCF calculations!!!
            train_sb_data,
            dev_sb_data,
            test_sb_data,
            augmentations_par={
                "amp_norm_type": "std",
                "window_strategy": "move",  # "pad"
                "final_windowlength": 3001,
                "sigma": 10,
                "fp_stabilization": 400,
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
                },
            batch_size=128,
            num_workers=24,
            random_seed=42):

        """ Modulus to prepare and process the data """
        self.augmentations_par = augmentations_par
        self.train_generator = sbg.GenericGenerator(train_sb_data)
        self.dev_generator = sbg.GenericGenerator(dev_sb_data)
        self.test_generator = sbg.GenericGenerator(test_sb_data)
        self.random_seed = random_seed
        self.train_loader, self.dev_loader, self.test_loader = None, None, None
        self.__training_epochs__ = None

        # ----------  0. Define query windows
        self.trainmod = pninstance
        self.model_epochs_list = []

        # ---------  1. Define augmentations
        self.augmentations = self.__define_augmentations__(**self.augmentations_par)

        # ---------  2. Load AUGMENTATIONS
        self.train_generator.add_augmentations(self.augmentations)
        self.dev_generator.add_augmentations(self.augmentations)
        self.test_generator.add_augmentations(self.augmentations)

        # ---------  3. Create DATALOADER
        self.train_loader = DataLoader(self.train_generator, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       worker_init_fn=self.__worker_init_fn_seed__)
        self.dev_loader = DataLoader(self.dev_generator, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     worker_init_fn=self.__worker_init_fn_seed__)
        self.test_loader = DataLoader(self.test_generator, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      worker_init_fn=self.__worker_init_fn_seed__)

    def set_random_seed(self, rndseed):
        self.random_seed = rndseed

    def __worker_init_fn_seed__(self, wid):
        np.random.seed(self.random_seed)

    def __worker_init_fn_full_seed__(self, wid):
        """ Just know, no chanche to modify RanomWindow seed number inside """
        def seed_everything(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        #
        seed_everything(self.random_seed)

    def __define_augmentations__(self, **kwargs):
        """ Define which augmentations to use in the training class """
        # samples_before = int(2 * kwargs["final_windowlength"] / 3)
        # windowlen = samples_before + int(2 * kwargs["final_windowlength"] / 3)
        # rw_windowlen = kwargs["final_windowlength"]
        # rw_low = 0
        # rw_high = windowlen

        samples_before = int(2 * kwargs["final_windowlength"] / 3) + kwargs["fp_stabilization"]
        windowlen = samples_before + int(2 * kwargs["final_windowlength"] / 3)
        rw_windowlen = kwargs["final_windowlength"] + kwargs["fp_stabilization"]
        rw_low = 0
        rw_high = windowlen

        augmentations_list = [
                sbg.WindowAroundSample(list(kwargs["phase_dict"].keys()),
                                       samples_before=windowlen/2,
                                       windowlen=windowlen,
                                       selection="first",
                                       strategy=kwargs["window_strategy"]),

                sbg.RandomWindow(windowlen=rw_windowlen,
                                 low=rw_low, high=rw_high,
                                 strategy=kwargs["window_strategy"]),

                sbg.FixedWindow(windowlen=kwargs["final_windowlength"],
                                p0=kwargs["fp_stabilization"],
                                strategy=kwargs["window_strategy"]),

                sbg.Normalize(demean_axis=-1,
                              amp_norm_axis=-1,
                              amp_norm_type=kwargs["amp_norm_type"]),

                sbg.ProbabilisticLabeller(label_columns=kwargs["phase_dict"],
                                          sigma=kwargs["sigma"], dim=0),

                sbg.ChangeDtype(np.float32, key="X"),
                sbg.ChangeDtype(np.float32, key="y")
            ]
        #
        return augmentations_list

    def extract_windows_cfs(self):
        outdict = {}

        for (gen, gen_name) in ((self.train_generator, "train"),
                                (self.dev_generator, "dev")):

            outdict[gen_name+"_X"], outdict[gen_name+"_Y"] = [], []

            for xx in tqdm(range(len(gen))):
                outdict[gen_name+"_X"].append(gen[xx]["X"])
                outdict[gen_name+"_Y"].append(gen[xx]["y"])
        #
        return outdict

    def __loss_fn__(self, y_pred, y_true, eps=1e-5):
        # vector cross entropy loss
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def __train_loop__(self, optimizer):
        size = len(self.train_loader.dataset)
        train_loss = 0
        train_loss_batches = []
        for batch_id, batch in enumerate(self.train_loader):
            # Compute prediction and loss
            pred = self.trainmod(batch["X"].to(
                                        self.trainmod.device))
            loss = self.__loss_fn__(pred, batch["y"].to(
                                        self.trainmod.device))
            train_loss_batches.append(loss.item())
            train_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                loss_val, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return (loss.item(), train_loss_batches)

    def __test_loop__(self):

        num_batches = len(self.dev_loader)
        test_loss = 0
        test_loss_batches = []

        self.trainmod.eval()  # 20230524

        with torch.no_grad():
            for batch in self.dev_loader:
                pred = self.trainmod(batch["X"].to(
                                self.trainmod.device))
                _tlb = self.__loss_fn__(pred, batch["y"].to(
                                self.trainmod.device)).item()

                test_loss += _tlb
                test_loss_batches.append(_tlb)

        self.trainmod.train()  # 20230524

        test_loss /= num_batches
        print(f"Dev. avg loss: {test_loss:>8f}\n")
        return (test_loss, test_loss_batches)

    def train_me(self,
                 # Train related
                 epochs=15,
                 optimizer_type="adam",
                 learning_rate=1e-2):

        """ Daje """

        # Defining OPTIMIZER
        if optimizer_type.lower() in ("adam", "adm"):
            optim = torch.optim.Adam(self.trainmod.parameters(),
                                     lr=learning_rate)
        else:
            raise ValueError("At the moment only the 'Adam' optimizer "
                             "is supported!")

        # ------------------------ GO
        train_loss_epochs, train_loss_epochs_batches = [], []
        test_loss_epochs, test_loss_epochs_batches = [], []

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            (_train_loss, _train_loss_batches) = self.__train_loop__(
                                                            optimizer=optim)
            train_loss_epochs.append(_train_loss)
            train_loss_epochs_batches.append(_train_loss_batches)

            (_test_loss, _test_loss_batches) = self.__test_loop__()
            test_loss_epochs.append(_test_loss)
            test_loss_epochs_batches.append(_test_loss_batches)
        #
        self.__training_epochs__ = t+1
        return (train_loss_epochs, train_loss_epochs_batches,
                test_loss_epochs, test_loss_epochs_batches)

    def train_me_early_stop(
                self,
                epochs=15,
                optimizer_type="adam",
                learning_rate=1e-2,
                patience=5,
                delta=0.001):  # Threshold for minimum improvement

        """ Daje """

        # Defining OPTIMIZER
        if optimizer_type.lower() in ("adam", "adm"):
            optim = torch.optim.Adam(self.trainmod.parameters(),
                                     lr=learning_rate)
        else:
            raise ValueError("At the moment only the 'Adam' optimizer "
                             "is supported!")

        # ------------------------ GO
        train_loss_epochs, train_loss_epochs_batches = [], []
        test_loss_epochs, test_loss_epochs_batches = [], []

        losses = []  # track losses for the last 'patience' epochs

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            (_train_loss, _train_loss_batches) = self.__train_loop__(
                                                            optimizer=optim)
            train_loss_epochs.append(_train_loss)
            train_loss_epochs_batches.append(_train_loss_batches)

            (_test_loss, _test_loss_batches) = self.__test_loop__()
            test_loss_epochs.append(_test_loss)
            test_loss_epochs_batches.append(_test_loss_batches)

            losses.append(_test_loss)

            # Store model for HISTORY porpuses
            self.model_epochs_list.append(
                copy.deepcopy(
                    self.trainmod))

            # If we have more than '2*patience' epochs done,
            # we can start checking for early stopping

            if t >= 2 * patience:
                (_answer, _, _step_back) = early_stop_criteria(
                                train_loss_epochs, train_loss_epochs_batches,
                                test_loss_epochs, test_loss_epochs_batches,
                                patience, delta)

                if _answer:
                    # Early stop triggered..pack everything
                    self.__training_epochs__ = (t+1) - _step_back
                    self.trainmod = copy.deepcopy(
                                        self.model_epochs_list[-_step_back-1])
                    del self.model_epochs_list
                    return (train_loss_epochs[:-_step_back],
                            train_loss_epochs_batches[:-_step_back],
                            test_loss_epochs[:-_step_back],
                            test_loss_epochs_batches[:-_step_back])

                else:
                    # Free unnecessary memory
                    if len(self.model_epochs_list) > patience:
                        del self.model_epochs_list[:-patience-1]

        # If here, we reached the maximum epochs provided by the user,
        # Return everything in full
        print("@@@ Reached the FULL  %d  epochs!" % epochs)
        self.__training_epochs__ = t+1
        return (train_loss_epochs, train_loss_epochs_batches,
                test_loss_epochs, test_loss_epochs_batches)

    def store_weigths(self, dir_path, model_name, jsonstring, version="1"):
        """ Store the finals """

        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)

        def _create_json_(docs, pathfile):
            with open(str(pathfile), "w") as OUT:

                OUT.write("{"+os.linesep)
                OUT.write(("    \"docstring\": \"%s\","+os.linesep) % docs)
                OUT.write("    \"model_args\": {"+os.linesep)
                OUT.write("        \"component_order\": \"ZNE\","+os.linesep)
                OUT.write("        \"phases\": \"PSN\""+os.linesep)
                OUT.write("    },"+os.linesep)
                OUT.write("    \"seisbench_requirement\": \"0.3.0\","+os.linesep)
                OUT.write(("    \"version\": \"%s\","+os.linesep) % version)
                OUT.write("    \"default_args\": {"+os.linesep)
                OUT.write("        \"blinding\": ["+os.linesep)
                OUT.write("            250,"+os.linesep)
                OUT.write("            250"+os.linesep)
                OUT.write("        ]"+os.linesep)
                OUT.write("    }"+os.linesep)
                OUT.write("}"+os.linesep)
        #
        if not dir_path.is_dir:
            dir_path.mkdir()

        _create_json_(jsonstring, dir_path / (str(model_name) + ".json"))
        torch.save(self.trainmod.state_dict(),
                   str(dir_path / (str(model_name) + ".pt"))
                   )

    def get_model(self):
        return self.trainmod

    def get_generator(self):
        return (self.train_generator, self.dev_generator, self.test_generator)

    def get_loader(self):
        return (self.train_loader, self.dev_loader, self.test_loader)
