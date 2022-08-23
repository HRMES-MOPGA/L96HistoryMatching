# Install EXETER_MOGP

git clone -b devel https://github.com/BayesExeter/ExeterUQ_MOGP

# BuildEmulator.R

packages <- c('reticulate','pracma', 'invgamma', 'GenSA', 'far', 'fields', 'lhs', 'maps', 'mco', 'mvtnorm', 'ncdf4', 'parallel', 'shape', 'tensor', 'withr', 'loo','MASS')
sapply(packages, require, character.only = TRUE, quietly = TRUE)
source("/home/jovyan/ExeterUQ_MOGP/BuildEmulator/AutoLMcode.R")
source("/home/jovyan/ExeterUQ_MOGP/BuildEmulator/CustomPredict.R")

# HistoryMatching.R

source("ExeterUQ_MOGP/HistoryMatching/impLayoutplot.R")

# find path of mogp_emulator

pip show mog_emulator
