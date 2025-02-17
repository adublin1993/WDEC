#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:16:03 2025

@author: adublin
"""

import numpy as np
import pandas as pd

# Import the other relevant notebook(s):
import Tabulation_Plots_Saving_Files as other_functions

def split_periods_results_dat(omega, file_ext):
    """
    This function reads in the rotation rate and the results.dat file extension. It outputs 
    the list of all rotationally split mode periods (using the frequency spacing formula). 

    Important: This function requires the output of the tabulate_results_dat function created 
    previously. 
    
    Input: 
    - results.dat file extension (str)
    
    Output: 
    - Pandas dataframe containing (k,l,m) and all rotationally split periods (df)
    """
    
    # Read in the results.dat dataframe:
    file_ext = str(file_ext)
    df = other_functions.tabulate_results_dat(file_ext)
   
    # Here you will need to read in the l values and ckl values
    ell = df['l']
    k = df['k']
    ckl = df['ckl']
    df['Omega']=omega
    
    # Instatitate a new dataframe for the output
    split_per_df = pd.DataFrame()
    
    # Steps 1 & 2: Convert from observed periods to frequencies.
    # Then apply rotational splitting
    
    for index, row in df.iterrows():
        k = (row['k']).astype(int)
        ell = (row['l']).astype(int)
        ckl = row['ckl']
        wdec_period = row['Period (s)']
        central_freq = 1/wdec_period
        
        m_vals = np.arange(-ell, ell+0.1, 1)
        m_vals = m_vals.astype(int)
        
        rotationally_split_per_list = []

        for m in m_vals:

            delta_nu_klm = m * (1 - ckl) * omega
            rotationally_split_freqs = central_freq + delta_nu_klm
            rotationally_split_pers = 1/rotationally_split_freqs
            rotationally_split_per_list.append(rotationally_split_pers)

        for m, split_period in zip(m_vals, rotationally_split_per_list):
            
            split_per_df = pd.concat([split_per_df, pd.DataFrame({'k': [k], 
                             'l': [ell], 'm': [m], 'Omega': [omega], 
                             'WDEC Period (s)': [wdec_period],
                             'Rotationally Split Period (s)': [split_period]})])
            
    return split_per_df
