#!/bin/bash

RND="777"
BATCH="64"
LR="0.0010"   # Must be a %.04f FORMAT!

# =================================================================
# =================================================================
# =================================================================


source /opt/anaconda/etc/profile.d/conda.sh  # SERRA
conda activate sometools_SBupdated


# =================================================================
# =================================================================
# =================================================================


TRAINDATA="INSTANCE"
TESTDATA_INDOMAIN="INSTANCE"
TESTDATA_CROSSDOMAIN="ETHZ"
SIZES=("NANO3" "NANO2" "NANO1" "NANO" "MICRO" "TINY" "SMALL" "MEDIUM" "LARGE")  # INSTANCE
EPOCHS=("100" "100" "100" "100" "100" "100" "100" "100" "100")  # INSTANCE

# TRAINDATA="ETHZ"
# TESTDATA_INDOMAIN="ETHZ"
# TESTDATA_CROSSDOMAIN="INSTANCE"
# SIZES=("NANO3" "NANO2" "NANO1" "NANO" "MICRO" "TINY")  # ETHZ
# EPOCHS=("100" "100" "100" "100" "100" "100")  # ETHZ

# Loop over the array
length=${#SIZES[@]}  # Get the length of the array

for ((i=0; i<length; i++)); do
  DATASIZE=${SIZES[i]}
  EPOCHSNUM=${EPOCHS[i]}

  echo ""
  echo ""
  echo "... Working with ---> ${TRAINDATA}  /  ${DATASIZE} - ${EPOCHSNUM} Epochs"
  echo ""
  echo "... Training DKPN"
  ./ReTrain_DKPN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
                    -e ${EPOCHSNUM} -b ${BATCH} -l ${LR} \
                    -o DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                    --early_stop -x 5 -y 0.0001
  echo ""
  echo "... Training PhaseNet"
  ./ReTrain_PN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
                  -e ${EPOCHSNUM} -b ${BATCH} -l ${LR} \
                  -o PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                  --early_stop -x 5 -y 0.0001

  # --- In-Domain  TEST
  echo ""
  echo "... Test IN-DOMAIN"
  ./LoadEvaluate_NEW.py  -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_INDOMAIN} -s ${DATASIZE} \
                         -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TRAINDATA}_${DATASIZE}_02
  ./LoadEvaluate_NEW.py  -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_INDOMAIN} -s ${DATASIZE} \
                         -x 0.5 -y 0.5 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TRAINDATA}_${DATASIZE}_05

  # --- Cross-Domain  TEST
  echo ""
  echo "... Test CROSS-DOMAIN"
  ./LoadEvaluate_NEW.py  -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_CROSSDOMAIN} -s ${DATASIZE} \
                         -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TESTDATA_CROSSDOMAIN}_${DATASIZE}_02
  ./LoadEvaluate_NEW.py  -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_CROSSDOMAIN} -s ${DATASIZE} \
                         -x 0.5 -y 0.5 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TESTDATA_CROSSDOMAIN}_${DATASIZE}_05

done

echo ""
echo "DONE!"
