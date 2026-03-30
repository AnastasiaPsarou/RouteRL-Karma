#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

REFERENCE_VALUES = [27.8686, 27.7669, 28.0364, 27.9816, 28.0983, 27.7383, 28.046, 27.7788, 27.8758, 27.8357, 27.7985, 27.8058]
REFERENCE_LABEL = "System Optimum"
REFERENCE_COLOR = "black"


USER_EQUILIBRIUM = None#[135.74, 136.10, 135.29, 135.61, 133.60]
REFERENCE_LABEL_2 = "User equilibrium"
"""ALGO_RUNS = {

    "fee 100": [
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_100/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_100/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_100/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_100/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_100/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_100/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_100/episodes",

    ],
    "monetary 15 norm":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_15_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_15_norm_ff/episodes",
    ],
    "monetary 50 norm":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_50_norm_ff/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_50_norm_ff/episodes",
    ],
    "monetary 50 norn greedy":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_50_norm_ff_greedy/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_50_norm_ff_greedy/episodes",
    ],
    "karma fee 5": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5/episodes",
    ],
    "karma fee 5 longer": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_longer/episodes",
    ],
    "karma fee 5 longer norm": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_norm/episodes",
    ],
    "karma fee 6": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_6/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_6/episodes",
    ],
            "karma fee 7": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_7_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_7_norm/episodes",
    ],
    "karma fee 8": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_8_norm/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_8_norm/episodes",
    ],


}"""

""" "monetary no dist cost":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_no_dist_cost/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_no_dist_cost/episodes",
    ],

    "karma norm reward": [
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_norm_reward/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_norm_reward/episodes",       
    ],"""

"""ALGO_RUNS = {
    
   
     "monetary pricing fee 20": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_20_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_20_heart_exps/episodes",
    ],

        "monetary pricing fee 10": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_10_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_10_heart_exps/episodes",
    ],

        "monetary pricing fee 5": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_5_heart_exps/episodes",
    ],
    
        "monetary pricing fee 7": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_7_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_7_heart_exps/episodes",
    ],

        "monetary pricing fee 6": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_6_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_6_heart_exps/episodes",
    ],

        "monetary pricing fee 4": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_4_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_4_heart_exps/episodes",
    ],
        "monetary pricing fee 3": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_3_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_3_heart_exps/episodes",
    ],
        "monetary pricing fee 2": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_2_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_2_heart_exps/episodes",
    ],
        "monetary pricing fee 1": [
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_1_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_1_heart_exps/episodes",
    ],
    "monetary pricing fee 1 longer": [
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_1_heart_exps_longer/episodes",
    ],
    "monetary pricing fee 2.5 longer": [
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_2_5_heart_exps/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_2_5_heart_exps/episodes",
    ],


}"""

ALGO_RUNS = {
    "monetary_pricing":[
       r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_heart_exps_same_st/episodes",
    ],
    "karma pricing":[
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_norm_reward_same_stt/episodes",
    ],
     "monetary pricing fee 1 longer": [
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_1_heart_exps_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_1_heart_exps_longer/episodes",
    ],
    "fee route 1":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_heart_exps_price_route_1_0_5/episodes",
    ],
    "fee route 1 1.5":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_1_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_1_5/episodes",
    ],
    "fee route 1 2":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_2/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_2/episodes",
    ],
    "fee route 1 5":[
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_route1_fee_5/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_route1_fee_5/episodes",
    ],
}


"""    "karma fee 4": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_longer/episodes",
    ],

    
    "monetary fee 0.5":[
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_0_5_vot_reward_longer/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_0_5_vot_reward_longer/episodes",


    ], """

""" 
 "karma fee 3_5 no tt": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price__3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_3_5_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_3_5_no_tt/episodes",
    ],

    "karma fee 8 no tt": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_8_no_tt/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_8_no_tt/episodes",
    ],
    "karma fee 4 p 0.5": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_p_0_5/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_p_0_5/episodes",
    ],

        "karma fee 4 p 0.2": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_16_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_17_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_18_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_19_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_20_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_21_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_22_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_23_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_24_karma_pricing_minimum_price_4_p_0_2/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_25_karma_pricing_minimum_price_4_p_0_2/episodes",
    ],"""

"""    "monetary fee 48": [
        r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_16_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_17_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_18_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_19_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_20_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_21_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_22_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_23_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_24_monetary_pricing_fee_48_no_urgency_longer_training/episodes",
        r"training_records_hyperippo_mlp_300_agents_seed_25_monetary_pricing_fee_48_no_urgency_longer_training/episodes",

    ],"""
"""

    "monetary pricing 47": [
            r"training_records_hyperippo_mlp_300_agents_seed_9_monetary_pricing_fee_47_no_urgency/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_10_monetary_pricing_fee_47_no_urgency/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_11_monetary_pricing_fee_47_no_urgency/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_12_monetary_pricing_fee_47_no_urgency/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_13_monetary_pricing_fee_47_no_urgency/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_14_monetary_pricing_fee_47_no_urgency/episodes",
            r"training_records_hyperippo_mlp_300_agents_seed_15_monetary_pricing_fee_47_no_urgency/episodes",
    ],
    
    
         "karma fee 5 longer": [
        r"training_records_hyperippo_masked_300_agents_seed_9_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_10_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_11_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_12_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_13_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_14_karma_pricing_minimum_price_5_longer/episodes",
        r"training_records_hyperippo_masked_300_agents_seed_15_karma_pricing_minimum_price_5_longer/episodes",

    ],"""
"""    "UE ": [
        r"scenarios/training_records_user_equilibrium_300_agents_9/episodes",
        r"training_records_user_equilibrium_300_agents_10/episodes",
        r"training_records_user_equilibrium_300_agents_11/episodes",
        r"training_records_user_equilibrium_300_agents_12/episodes",
        r"training_records_user_equilibrium_300_agents_13/episodes",
    ],"""

"""ALGO_RUNS = {
    "UE ": [
        r"scenarios/training_records_user_equilibrium_300_agents_9/episodes",
        r"training_records_user_equilibrium_300_agents_10/episodes",
        r"training_records_user_equilibrium_300_agents_11/episodes",
        r"training_records_user_equilibrium_300_agents_12/episodes",
        r"training_records_user_equilibrium_300_agents_13/episodes",
    ],
    "Monetary pricing":[
        r"scenarios/monetary_pricing/training_records_monetary_pricing_300_agents_9_fee_10/episodes",
    ]
}"""


"""ALGO_RUNS = {

    "Monetary 20":[
        r"training_records_monetary_pricing_300_agents_9_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_20/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_20/episodes",
    ],

    "Karma 8": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_8/episodes",
    ],
    "UE ": [
        #r"training_records_user_equilibrium_300_agents_9_40/episodes",
        r"training_records_user_equilibrium_300_agents_10_50/episodes",
        r"training_records_user_equilibrium_300_agents_11_50/episodes",
        r"training_records_user_equilibrium_300_agents_12_50/episodes",
    ],
}"""
"""ALGO_RUNS = {
    "Monetary 5":[
        r"training_records_monetary_pricing_300_agents_9_fee_5_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_5_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_5_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_5_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_5_no_urgency/episodes",
    ],

    "Monetary 10":[
        r"training_records_monetary_pricing_300_agents_9_fee_10_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_10_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_10_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_10_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_10_no_urgency/episodes",
    ],

    "Monetary 12":[
        r"training_records_monetary_pricing_300_agents_9_fee_12_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_12_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_12_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_12_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_12_no_urgency/episodes",
    ],

    "Monetary 15":[
        r"training_records_monetary_pricing_300_agents_9_fee_15_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_15_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_15_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_15_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_15_no_urgency/episodes",
    ],


    "Monetary 20":[
        r"training_records_monetary_pricing_300_agents_9_fee_20_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_10_fee_20_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_11_fee_20_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_12_fee_20_no_urgency/episodes",
        r"training_records_monetary_pricing_300_agents_13_fee_20_no_urgency/episodes",
    ],


    "Karma 8": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_8/episodes",
    ],

    "Karma 9": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_9/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_9/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_9/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_9/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_9/episodes",
    ],
    "UE ": [
        #r"training_records_user_equilibrium_300_agents_9_40/episodes",
        r"training_records_user_equilibrium_300_agents_10_50/episodes",
        r"training_records_user_equilibrium_300_agents_11_50/episodes",
        r"training_records_user_equilibrium_300_agents_12_50/episodes",
    ],
}"""

"""    "Karma 6": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_6/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_6/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_6/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_6/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_6/episodes",
        r"training_records_karma_300_agents_14_mean_field_new_net_centr_price_6/episodes",
    ],
    "Karma 7": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_7/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_7/episodes",
    ],
    "Karma 8": [
        r"training_records_karma_300_agents_9_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_10_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_11_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_12_mean_field_new_net_centr_price_8/episodes",
        r"training_records_karma_300_agents_13_mean_field_new_net_centr_price_8/episodes",
    ],
    "UE ": [
        #r"training_records_user_equilibrium_300_agents_9_40/episodes",
        r"training_records_user_equilibrium_300_agents_10_50/episodes",
        r"training_records_user_equilibrium_300_agents_11_50/episodes",
        r"training_records_user_equilibrium_300_agents_12_50/episodes",
    ],"""

PLOT_STAT = "travel_time"     # column in CSV
KIND_COL = "kind"             # filter column (if present)
PLOT_KIND = "AV"              # "AV", "Human", or "All"
SMOOTH = 20                   # moving average window; 0/1 disables
MAX_EPISODE = 2500            # None = no cap
XLIM = (0, MAX_EPISODE)       # None = auto
SAVE_PNG = "imgs/mean_travel_time_across_algos.png"  # None to skip saving
TITLE = f"Mean {PLOT_STAT}"
XLABEL = "Iterations (days)"
YLABEL = f"Mean {PLOT_STAT}"
FIGSIZE = (8.5, 4.0)
BAND_ALPHA = 0.18
PLOT_MEDIAN = False   # True = plot median (dashed), False = do not plot median
YLIM =None#(25, 32) #(20, 60)  


# If you want explicit colors per algorithm, set them here; otherwise matplotlib cycles.
ALGO_COLORS = {
    "Karma": "firebrick",
    "Monetary pricing": "teal",
}

"""
"firebrick", "teal", "peru", "navy", 
        "salmon", "slategray", "darkviolet", 
        "lightskyblue", "darkolivegreen", "black"
"""

MEDIAN_LS = "--"
MEDIAN_LW = 1.6
MEAN_LW = 2.0

# ==========================================================
def smooth_ma(y, w):
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1 or len(y) < w:
        return y
    half = w // 2
    pad = np.pad(y, (half, half), mode="edge")
    out = np.convolve(pad, np.ones(w) / w, mode="valid")
    return out[:len(y)]

def get_episode_files(folder):
    if not os.path.isdir(folder):
        print(f"[WARN] not a folder: {folder}")
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.startswith("ep") and fn.endswith(".csv"):
            try:
                ep = int(fn[2:-4])
                files.append((ep, fn))
            except ValueError:
                pass
    files.sort(key=lambda t: t[0])
    if MAX_EPISODE is not None:
        files = [(ep, fn) for ep, fn in files if ep <= MAX_EPISODE]
    return files

def per_episode_mean_for_folder(folder, metric=PLOT_STAT, kind=PLOT_KIND, kind_col=KIND_COL):
    """
    Returns DataFrame with columns: episode, mean
    Mean is across rows/agents within each episode-file.
    """
    rows = []
    for ep, fn in get_episode_files(folder):
        path = os.path.join(folder, fn)
        if not os.path.getsize(path):
            continue
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] read failed {path}: {e}")
            continue

        if metric not in df.columns:
            continue

        if kind != "All" and kind_col in df.columns:
            df = df[df[kind_col] == kind]

        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) == 0:
            continue

        rows.append((ep, float(vals.mean())))

    if not rows:
        return pd.DataFrame(columns=["episode", "mean"])

    out = (
        pd.DataFrame(rows, columns=["episode", "mean"])
        .sort_values("episode")
        .reset_index(drop=True)
    )
    return out

def align_and_aggregate_across_folders(series_list):
    """
    Input: list of (label, df) where df has columns ['episode','mean'] for each folder/replication.
    Output: DataFrame with columns: episode, mean_across_rep, median_across_rep, std_across_rep, n_rep
    """
    all_eps = sorted(
        set().union(*[set(df["episode"]) for _, df in series_list if not df.empty])
    )
    if not all_eps:
        return pd.DataFrame(
            columns=["episode", "mean_across_rep", "median_across_rep", "std_across_rep", "n_rep"]
        )

    Y = np.full((len(series_list), len(all_eps)), np.nan, dtype=float)

    for r, (_, df) in enumerate(series_list):
        if df.empty:
            continue
        m = dict(zip(df["episode"].to_numpy(), df["mean"].to_numpy()))
        for c, ep in enumerate(all_eps):
            if ep in m:
                Y[r, c] = m[ep]

    n_rep = np.sum(~np.isnan(Y), axis=0)
    mean_across = np.nanmean(Y, axis=0)
    median_across = np.nanmedian(Y, axis=0) if PLOT_MEDIAN else None

    std_across = np.array(
        [
            np.nanstd(Y[:, i], ddof=1) if n_rep[i] >= 2 else (0.0 if n_rep[i] == 1 else np.nan)
            for i in range(len(all_eps))
        ],
        dtype=float,
    )

    data = {
        "episode": all_eps,
        "mean_across_rep": mean_across,
        "std_across_rep": std_across,
        "n_rep": n_rep,
    }

    if PLOT_MEDIAN:
        data["median_across_rep"] = median_across

    out = pd.DataFrame(data)

    return out

def load_and_aggregate_for_algorithm(algo_name, folders):
    """
    Build per-folder per-episode means for one algorithm, then aggregate across its replications.
    Returns (algo_name, agg_df, max_n_rep)
    """
    series = []
    for folder in folders:
        print(f"[INFO] {algo_name}: loading {folder}")
        df = per_episode_mean_for_folder(folder)
        if df.empty:
            print(f"[WARN] {algo_name}: no usable episodes in {folder} for kind='{PLOT_KIND}'")
        label = os.path.basename(folder) or folder
        series.append((label, df))

    agg = align_and_aggregate_across_folders(series)
    if agg.empty:
        print(f"[WARN] {algo_name}: no usable data across replications.")
        return algo_name, agg, 0

    return algo_name, agg, int(np.nanmax(agg["n_rep"].to_numpy()))

def main():
    # 1) Aggregate each algorithm separately
    algo_aggs = []
    for algo_name, folders in ALGO_RUNS.items():
        algo_aggs.append(load_and_aggregate_for_algorithm(algo_name, folders))

    # Keep only algorithms with some data
    algo_aggs = [(name, agg, nmax) for (name, agg, nmax) in algo_aggs if not agg.empty]
    if not algo_aggs:
        raise ValueError("No usable data for any algorithm. Check paths/kind/MAX_EPISODE.")

    # 2) Plot: one mean+std band + median per algorithm
    plt.figure(figsize=FIGSIZE)

    for algo_name, agg, nmax in algo_aggs:
        x = agg["episode"].to_numpy()
        y = agg["mean_across_rep"].to_numpy(dtype=float)
        s = agg["std_across_rep"].to_numpy(dtype=float)
        n = agg["n_rep"].to_numpy(dtype=int)

        # smoothing
        y_plot = smooth_ma(y, SMOOTH)
        s_plot = smooth_ma(s, SMOOTH)

        if PLOT_MEDIAN:
            med = agg["median_across_rep"].to_numpy(dtype=float)
            med_plot = smooth_ma(med, SMOOTH)



        color = ALGO_COLORS.get(algo_name, None)

        # mean line
        (line,) = plt.plot(
            x, y_plot,
            lw=MEAN_LW,
            color=color,
            label=f"{algo_name} (n={nmax})",
        )

        line_color = line.get_color()

        # std band (same color as line)
        mask = n >= 2
        if np.any(mask):
            plt.fill_between(
                x[mask],
                (y_plot - s_plot)[mask],
                (y_plot + s_plot)[mask],
                color=line_color,
                alpha=BAND_ALPHA,
                linewidth=0,
            )

        # median line (same color, dashed)
        if PLOT_MEDIAN:
            plt.plot(
                x, med_plot,
                linestyle=MEDIAN_LS,
                linewidth=MEDIAN_LW,
                color=color,
                label=f"{algo_name} median",
            )

    if XLIM is not None:
        plt.xlim(*XLIM)

    if YLIM is not None:
        plt.ylim(*YLIM)

        # --------------------------------------------------
    # Add horizontal reference mean + std band
    # --------------------------------------------------
    if REFERENCE_VALUES:
        vals = np.asarray(REFERENCE_VALUES, dtype=float)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals, ddof=1)

        x_ref = np.array([0, MAX_EPISODE], dtype=float)

        # mean line
        (ref_line,) = plt.plot(
            x_ref,
            [mu, mu],
            lw=MEAN_LW,
            color=REFERENCE_COLOR,
            linestyle=":",
            label=f"{REFERENCE_LABEL} (mean)",
        )

        # std band
        if len(vals) >= 2:
            plt.fill_between(
                x_ref,
                [mu - sigma, mu - sigma],
                [mu + sigma, mu + sigma],
                color=REFERENCE_COLOR,
                alpha=BAND_ALPHA,
                linewidth=0,
            )

    if USER_EQUILIBRIUM:
        vals = np.asarray(USER_EQUILIBRIUM, dtype=float)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals, ddof=1)

        x_ref = np.array([0, MAX_EPISODE], dtype=float)

        # mean line
        (ref_line,) = plt.plot(
            x_ref,
            [mu, mu],
            lw=MEAN_LW,
            color=REFERENCE_COLOR,
            linestyle=":",
            label=f"{REFERENCE_LABEL_2} (mean)",
        )

        # std band
        if len(vals) >= 2:
            plt.fill_between(
                x_ref,
                [mu - sigma, mu - sigma],
                [mu + sigma, mu + sigma],
                color=REFERENCE_COLOR,
                alpha=BAND_ALPHA,
                linewidth=0,
            )


    plt.xlabel(XLABEL, fontsize=20)
    plt.ylabel(YLABEL, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(which="both")
    plt.title(TITLE, fontsize=14)
    plt.legend(fontsize=11, loc="best", ncol=2)
    plt.tight_layout()

    if SAVE_PNG:
        os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
        plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
        print(f"[INFO] saved plot to: {SAVE_PNG}")

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
