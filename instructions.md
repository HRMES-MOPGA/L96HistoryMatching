# install mogp_emulator

pip install mogp_emulator=0.5.0

# find path of mogp_emulator

pip show mogp_emulator

# Install EXETER_MOGP

git clone -b devel https://github.com/BayesExeter/ExeterUQ_MOGP

# BuildEmulator.R

source("ExeterUQ_MOGP/BuildEmulator/AutoLMcode.R")
source("ExeterUQ_MOGP/BuildEmulator/CustomPredict.R")

# HistoryMatching.R

source("ExeterUQ_MOGP/HistoryMatching/impLayoutplot.R")

