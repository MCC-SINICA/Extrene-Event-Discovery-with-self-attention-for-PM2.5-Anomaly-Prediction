# Extrene-Event-Discovery-with-self-attention-for-PM2.5-Anomaly-Prediction
This repository store the code for Extrene Event Discovery with self attention for PM2.5 Anomaly Prediction
## Please use the package version in the requirements file to run the program.

## Contains folders:
checkpoints : saving the model checkpoints during training.

configs : saving the model setting during training.

data : After pre-processing

Final_Results : The results which showed in paper

results : The results which produced by test.py


## Contains files:
argument.py

constants.py

custom_loss.py

dataset.py

extreme.py

model.py

networks.py

preprocess.py

requirements.txt

test.py

utils.py

## How to use

Under the folder and type the command on the terminal, To produce the results of Table 3, XFMR_EVL 

P : 0.1560, R : 0.5432, F1 : 0.2387, MCC : 0.2594

Use the following command 

python test.py --no 281 --device 0

## Remark

To view the results which showed in paper, plz check the folder "Final_Results"

281 = XFMR_EVL

477 = DMXFMR_EVL

478 = LMXFMR_EVL





## Ongoing

since the chickpoints of the model too many, we continue upload it.
