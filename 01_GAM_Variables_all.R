# Project: HSI_HT-EADs
# Project start time: 2024/01/12
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to test the GAM model for all heatstroke, with raw climate variables as inputs


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

jp_clm_mean_summer <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_clm_mean_summer"]]  # (6, 47, 1220) ''dewt', 'at', 'rad', 'ws', 'pres', 'rh'
jp_clm_mean_summer_1 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_clm_mean_summer_1"]]  # (6, 47, 1220) ''dewt', 'at', 'rad', 'ws', 'pres', 'rh'
jp_clm_mean_summer_2 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_clm_mean_summer_2"]]  # (6, 47, 1220) ''dewt', 'at', 'rad', 'ws', 'pres', 'rh'

# load the output data
all_htk <- np$load(paste0(input_dir, "model_input.npz"))$f[["all_htk"]]  # (47, 1220)

# pre-define the array for the predictions
all_prediction <- array(NA, dim=c(5, 10, 47, 1220))

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

  
  # for multiple climate variables

  pf_data$tair <- jp_clm_mean_summer[2, pf,]
  pf_data$rh <- jp_clm_mean_summer[6, pf,]
  pf_data$ws <- jp_clm_mean_summer[4, pf,]
  pf_data$rad <- jp_clm_mean_summer[3, pf,]
  pf_data$dewt <- jp_clm_mean_summer[1, pf,]
  pf_data$pres <- jp_clm_mean_summer[5, pf,]
  
  pf_data$tair_1 <- jp_clm_mean_summer_1[2, pf,]
  pf_data$rh_1 <- jp_clm_mean_summer_1[6, pf,]
  pf_data$ws_1 <- jp_clm_mean_summer_1[4, pf,]
  pf_data$rad_1 <- jp_clm_mean_summer_1[3, pf,]
  pf_data$dewt_1 <- jp_clm_mean_summer_1[1, pf,]
  pf_data$pres_1 <- jp_clm_mean_summer_1[5, pf,]
  
  pf_data$tair_2 <- jp_clm_mean_summer_2[2, pf,]
  pf_data$rh_2 <- jp_clm_mean_summer_2[6, pf,]
  pf_data$ws_2 <- jp_clm_mean_summer_2[4, pf,]
  pf_data$rad_2 <- jp_clm_mean_summer_2[3, pf,]
  pf_data$dewt_2 <- jp_clm_mean_summer_2[1, pf,]
  pf_data$pres_2 <- jp_clm_mean_summer_2[5, pf,]
  
  
  # cross-validation
  for (yy in seq(10)){
    
    vali_idx <- which(date_array_summer[,1] == yy + 2009)
    cali_idx <- which(date_array_summer[, 1] != yy + 2009)
    
    
    for (var in seq(5)){
      
      if (var == 1){
        # 1 var
        mod_clm = gam(heatstroke ~ s(tair, k=k_sum)
                      + s(tair_1, k=k_sum)
                      + s(tair_2, k=k_sum)
                      + dfg + s(old_prop, k=3) + factor(holiday) + factor(dow)  + offset(log(population)), 
                      data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
        
      }else if (var == 2){
        # 2 vars
        mod_clm = gam(heatstroke ~ s(tair, k=k_sum) + s(rh, k=k_sum)
                      + s(tair_1, k=k_sum) + s(rh_1, k=k_sum)
                      + s(tair_2, k=k_sum) + s(rh_2, k=k_sum)
                      + dfg + s(old_prop, k=3) + factor(holiday) + factor(dow)  + offset(log(population)), 
                      data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
        
      }else if (var == 3){
        # 3 vars
        mod_clm = gam(heatstroke ~ s(tair, k=k_sum) + s(rh, k=k_sum) + s(ws, k=k_sum)
                      + s(tair_1, k=k_sum) + s(rh_1, k=k_sum) + s(ws_1, k=k_sum)
                      + s(tair_2, k=k_sum) + s(rh_2, k=k_sum) + s(ws_2, k=k_sum)
                      + dfg + s(old_prop, k=3) + factor(holiday) + factor(dow)  + offset(log(population)), 
                      data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
        
      }else if (var == 4){
        # 4 vars
        mod_clm = gam(heatstroke ~ s(tair, k=k_sum) + s(rh, k=k_sum) + s(ws, k=k_sum) + s(rad, k=k_sum)
                            + s(tair_1, k=k_sum) + s(rh_1, k=k_sum) + s(ws_1, k=k_sum) + s(rad_1, k=k_sum)
                            + s(tair_2, k=k_sum) + s(rh_2, k=k_sum) + s(ws_2, k=k_sum) + s(rad_2, k=k_sum)
                            + dfg + s(old_prop, k=3) + factor(holiday) + factor(dow)  + offset(log(population)), 
                            data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
      }else if (var == 5){
        # 5 original vars
        mod_clm = gam(heatstroke ~ s(tair, k=k_sum) + s(dewt, k=k_sum) + s(rad, k=k_sum) + s(ws, k=k_sum) + s(pres, k=k_sum)
                      + s(tair_1, k=k_sum) + s(dewt_1, k=k_sum) + s(rad_1, k=k_sum) + s(ws_1, k=k_sum) + s(pres_1, k=k_sum) 
                      + s(tair_2, k=k_sum) + s(dewt_2, k=k_sum) + s(rad_2, k=k_sum) + s(ws_2, k=k_sum) + s(pres_2, k=k_sum) 
                      + dfg + s(old_prop, k=3) + factor(holiday) + factor(dow)  + offset(log(population)), 
                      data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
      }
  
    
    all_prediction[var, yy, pf, cali_idx] <- predict(mod_clm, newdata = pf_data[cali_idx,], type = "response")
    all_prediction[var, yy, pf, vali_idx] <- predict(mod_clm, newdata = pf_data[vali_idx,], type = "response")
    
    print(paste0("Prefecture ", pf, " CLM Year ", yy + 2009, " Var ", var, " Finished."))
    
    }
  
  }

}

save(all_prediction,
     file=paste0(input_dir, "CLM_all_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata"))

print("All Finished.")







