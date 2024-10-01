# Project: HSI_HT-EADs
# Project start time: 2024/01/12
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to test the GAM model for all heatstroke, with different HSIs as inputs


rm(list=ls())

# load libraries
library(reticulate)
library(lubridate)
library(MASS)
library(ggplot2)
library(patchwork)
library(mgcv)

np <- import("numpy")
input_dir <- "C:/Users/gq646/Desktop/GAM/Information_Lose/"

# load the input data
date_array_summer <- np$load(paste0(input_dir, "model_input.npz"))$f[["date_array_summer"]]  # (1220, 6)
social_economic <- np$load(paste0(input_dir, "model_input.npz"))$f[["social_economic"]]  # (47, 3, 1220)   population, old, income

jp_hsi_mean_summer <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_hsi_mean_summer"]]  # (9, 47, 1220) 'at', 'tw', 'ts', 'wbgt', 'swbgt', 'hx', 'apt', 'utci', 'hi' 
jp_hsi_mean_summer_1 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_hsi_mean_summer_1"]]  # (9, 47, 1220)  'at', 'tw', 'ts', 'wbgt', 'swbgt', 'hx', 'apt', 'utci', 'hi' 
jp_hsi_mean_summer_2 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_hsi_mean_summer_2"]]  # (9, 47, 1220) 'at', 'tw', 'ts', 'wbgt', 'swbgt', 'hx', 'apt', 'utci', 'hi' 

# load the output data
all_htk <- np$load(paste0(input_dir, "model_input.npz"))$f[["all_htk"]]  # (47, 1220)

# pre-define the array for the predictions
all_prediction <- array(NA, dim=c(9, 10, 47, 1220))

k_sum = 9
# Tokyo 13, Osaka 27, Aichi 23

for (pf in seq(47)){
  
  pf_data <- data.frame(
    year = date_array_summer[,1],
    month = date_array_summer[,2],
    day = date_array_summer[,3],
    dow = date_array_summer[,4],
    holiday = date_array_summer[,5],
    dfg = date_array_summer[,6],
    population = social_economic[pf, 1,],
    old_prop = social_economic[pf, 2,] / social_economic[pf, 1,], 
    heatstroke = all_htk[pf,]
    )

  
  # cross-validation
  for (yy in seq(10)){
    
    vali_idx <- which(date_array_summer[,1] == yy + 2009)
    cali_idx <- which(date_array_summer[, 1] != yy + 2009)
    
    
    for (var in seq(9)){
      
      pf_data$hsi <- jp_hsi_mean_summer[var, pf,]
      pf_data$hsi_1 <- jp_hsi_mean_summer_1[var, pf,]
      pf_data$hsi_2 <- jp_hsi_mean_summer_2[var, pf,]

      # HSI
      mod_clm = gam(heatstroke ~ s(hsi, k=k_sum)
                    + s(hsi_1, k=k_sum)
                    + s(hsi_2, k=k_sum)
                    + dfg + s(old_prop, k=3) + factor(holiday) + factor(dow)  + offset(log(population)), 
                    data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
        
    
    all_prediction[var, yy, pf, cali_idx] <- predict(mod_clm, newdata = pf_data[cali_idx,], type = "response")
    all_prediction[var, yy, pf, vali_idx] <- predict(mod_clm, newdata = pf_data[vali_idx,], type = "response")
    
    
    print(paste0("Prefecture ", pf, " HSI Year ", yy + 2009, " Var ", var, " Finished."))
    
    }
  
  }

}

save(all_prediction,
     file=paste0(input_dir, "HSI_all_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata"))

print("All Finished.")







