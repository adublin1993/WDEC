#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:21:23 2025

@author: adublin
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Import the other relevant notebook(s)
import Tabulation_Plots_Saving_Files as other_functions

# Update this as appropriate
global_bergeron_table_mass_file_dir = '/Users/adublin/Desktop/WDEC/Bergeron_DBV_grid/'
'Bergeron_Tables/'

def add_noise_to_model_periods(subset_of_model_periods, measured_period_uncertainties):
    """
    This function adds noise to a model period. The "model" period is the original (model)
    period. Any measured period is noisy. The measured period corresponds to *one* model
    period (fixed WDEC parameters) with noise.

    Input: 
    - array or list (subset of model periods)
    - array or list (measured period uncertainties). 
    
    Output: 
    - array (measured periods)
    """

    subset_of_model_periods = np.array(subset_of_model_periods)
    measured_period_uncertainties = np.array(measured_period_uncertainties)
    
    num_measured_periods = len(subset_of_model_periods)
    num_measured_uncertainties = len(measured_period_uncertainties)

    # Must enforce that the arrays have equal length
    
    if num_measured_periods != num_measured_uncertainties:
    
        raise ValueError("The number of measured period uncertainties does not match the number of \n"
            "observed periods.")
    
    else:
    
        # NB: Uncertainties should be positive
        measured_period_array = (subset_of_model_periods + \
            np.abs(measured_period_uncertainties) * np.abs(np.random.randn(num_measured_periods)))
        
        measured_period_array = np.round(measured_period_array, 3)

        return measured_period_array

def add_noise_to_model_abs_mag(model_absolute_magnitude, measured_abs_mag_uncertainty):
    """
    This function adds noise to a model (G-band, or other) absolute magnitude. Any measured absolute 
    magnitude (G-band, or other) is noisy. The measured absolute magnitude corresponds to *one* model
    absolute magnitude with noise (fixed WDEC parameters). Note that the uncertainty, of course, must 
    be positive.

    Input: 
    - float or int (model absolute magnitude)
    - float or int (model absolute magnitude uncertainty).
    
    Output: 
    - float or int (absolute magnitude with noise).
    
    """

    # Enforce that only one (i.e., zero-dimensional) numerical value is provided for either the
    # absolute magnitude or the corresponding uncertainty.
    
    mag = model_absolute_magnitude
    sig = measured_abs_mag_uncertainty

    if np.ndim(mag)!=0 and np.ndim(sig)!=0:

        raise ValueError(f"More than one value provided for model absolute magnitude: {mag} \n"
                         f"More than one value provided for measured absolute magnitude uncertainty: {sig}") 
    
    elif np.ndim(mag)!=0:

        raise ValueError(f"More than one value provided for model absolute magnitude: {mag} \n")

    elif np.ndim(sig)!=0: 
        
        raise ValueError(f"More than one value provided for measured absolute magnitude uncertainty: {sig} \n")
    
    else:    
        
        fiducial_mag = np.array([model_absolute_magnitude]) # np dimension one, used below
        
        measured_abs_mag = (model_absolute_magnitude + \
            np.abs(measured_abs_mag_uncertainty) * np.abs(float(np.random.randn(np.ndim(fiducial_mag)))))
    
        measured_abs_mag = np.round(measured_abs_mag, 3)

    return measured_abs_mag
    
def process_bergeron_file_one_mass(full_path_file_name, 
                                       names=["Teff", "log g", "Mbol", "BC", "U", \
                                              "B", "V", "R", "I", "J", "H", "Ks", \
                                              "Y", "J", "H", "K", "W1", "W2", "W3", \
                                              "W4", "S3.6", "S4.5", "S5.8", "S8.0", \
                                              "u", "g", "r", "i", "z", "g", "r", "i", \
                                              "z", "y", "G2", "G2_BP", "G2_RP", "G3", \
                                              "G3_BP", "G3_RP", "FUV", "NUV", "Age"]):
    """
    This function should be called on the full path of the specific "...Table_Mass_(...)"
    file.

    Input: 
        full path of the "...Table_Mass_(...)" file name (str)
        column names of the Bergeron "...Table_Mass_(...)" file (list)

    Output:
        full Bergeron table for the model of the specified mass (dataframe)
    """
    
    # Extract the model mass from the contents of the .txt file.
    
    with open(full_path_file_name, 'r') as f:

        lines = f.readlines()

        mass = lines[0].strip().split()[1]

        model_mass = 1000*float(mass)

    # With the model mass stored in memory, tabulate the data with the 
    # column names. (Skip the first row that contains erroneous text.)

    # The dataframe (df) originally has 44 columns (the column 'g' gets 
    # incorrectly read in as an additional separate column). The 
    # following code addresses this. 

    with open(full_path_file_name, 'r') as f:

        df = pd.read_csv(f, sep=r'\s+', skiprows=1, header=0) 

    # The "names" list has 43 elements (strings)
    if len(df.columns) > len(names): 

        # Now both have length 43, ignoring the last column of NaNs
        df = df.iloc[:, :-1] 

    df.columns = names[:len(df.columns)] 

    df.insert(1, 'Total Mass', model_mass)

    df.index = df.index + 1

    return df

def closest_bergeron_table_mass_dataframe(Mass):
    """
    Tabulate the Bergeron data for models whose mass is closest to the fiducial mass. 

    Input
        test model mass (float or int)

    Output
        Pandas dataframe (df)
    """
    if Mass < 0.15*1000:
        
        print("Fiducial model mass of {Mass/1000}M is considerably less than "
              "the minimum mass of 0.2M available in the DBV Bergeron " 
              "Data Table collection.")
        print("The Bergeron table for 0.2M will be used, but beware "
              "that this may introduce significant inaccuracies.")
        
        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                            'Table_Mass_0.2')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.2M")
        
        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df
    
    elif 0.15*1000 <= Mass < 0.25*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.2')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.2M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df
            
    elif 0.25*1000 <= Mass < 0.35*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.3')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.3M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.35*1000 <= Mass < 0.45*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.4')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.4M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.45*1000 <= Mass < 0.55*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.5')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.5M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.55*1000 <= Mass < 0.65*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.6')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.6M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.65*1000 <= Mass < 0.75*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.7')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.7M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.75*1000 <= Mass < 0.85*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.8')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.8M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.85*1000 <= Mass < 0.95*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.9')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_0.9M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 0.95*1000 <= Mass < 1.05*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.0')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_1.0M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 1.05*1000 <= Mass < 1.15*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.1')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_1.1M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 1.15*1000 <= Mass < 1.25*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.2')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_1.2M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df

    elif 1.25*1000 <= Mass < 1.35*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.3')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_1.3M")

        df = process_bergeron_file_one_mass(table_mass_file_name)

        # display(df)

        return df
    
    elif 1.35*1000 <= Mass < 1.40*1000:
        
        print("Approaching the Chandrasekhar limit. The data table for 1.3M will be " \
              "used. Beware that this may introduce significant inaccuracies.")
        
        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                            'Table_Mass_1.3')

        # print(f"Fiducial Mass: {Mass/1000}M")
        # print("Bergeron File: Table_Mass_1.3M")
        
        df = process_bergeron_file_one_mass(table_mass_file_name)
    
        # display(df)

        return df
        
    elif 1.40*1000 <= Mass:
        
        # print(f"Fiducial Mass: {Mass/1000}M")
        
        raise ValueError(f"Chandrasekhar limit reached. The fiducial model for {Mass/1000}M is "
                         f"unphysical.")

        return None

def full_bergeron_table_mass_filename(Mass):
    """
    This function is very similar to the previous one, except that we just
    need the file names and can skip everything else. It outputs the full
    file name (str) of the Bergeron models whose mass is closest to the 
    fiducial mass. It is helpful for future functions. 

    Input:
    - test model mass (float or int)
    
    Output:
    - full "...Table_Mass_(...)" file path (str)
    """
    
    if Mass < 0.15*1000:
    
        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                            'Table_Mass_0.2')

    elif 0.15*1000 <= Mass < 0.25*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.2')
            
    elif 0.25*1000 <= Mass < 0.35*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.3')

    elif 0.35*1000 <= Mass < 0.45*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.4')

    elif 0.45*1000 <= Mass < 0.55*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.5')

    elif 0.55*1000 <= Mass < 0.65*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.6')

    elif 0.65*1000 <= Mass < 0.75*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.7')

    elif 0.75*1000 <= Mass < 0.85*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.8')

    elif 0.85*1000 <= Mass < 0.95*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.9')

    elif 0.95*1000 <= Mass < 1.05*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.0')

    elif 1.05*1000 <= Mass < 1.15*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.1')

    elif 1.15*1000 <= Mass < 1.25*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.2')

    elif 1.25*1000 <= Mass < 1.35*1000:

        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.3')
    
    elif 1.35*1000 <= Mass < 1.40*1000:

        print("Approaching the Chandrasekhar limit. The data table for 1.3M will be " \
              "used.")
        
        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                            'Table_Mass_1.3')
        
    elif 1.40*1000 <= Mass:
        
        raise ValueError(f"Chandrasekhar limit reached. The fiducial model for {Mass/1000}M is "
                         f"unphysical.")

    # Return the appropriate file name:
    
    return table_mass_file_name
      
def interpolator_function(x, y):
    """
    Basic 1D interpolator function.

    Input:
    - x (array)
    - y (array)
    
    Output:
    - 1d interpolator function object
    """
    
    interpolator_fcn = interp1d(x, y)
    
    return interpolator_fcn
    
def bergeron_interpolation(Teff, Mass, Menv, 
                           Mhe, Mh, He_abund_mixed_CHeH_region,
                           diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                           alpha, h1x100, h2x100, h3x100, 
                           w1x100, w2x100, w3x100, w4x100, 
                           which_filter=None):
    """
    This function interpolates over the *Bergeron* effective temperatures 
    to calculate radii and/or absolute magnitudes in the specified bandpass. 
    Note that the input parameters correspond to those of the fiducial model.

    Input: 
        - all fiducial model parameters (ints and/or floats)
        - filter (str)
    
    Available filters (optional): 
    'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks', 'Y', 'J', 'H', 'K', 'W1', 'W2',     
    'W3', 'W4', 'S3.6', 'S4.5', 'S5.8', 'S8.0', 'u', 'g', 'r', 'i', 'z', 'g', 'r', 'i', 'z', 'y',
    'G2', 'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'
            
    Output: 
        - interpolated radius (float) and/or...
        - interpolated absolute magnitude (float)
    """
    
    # Constants: 
    
    G = 6.67259e-8 # cm3 g-1 s-2 
    Msun = 1.99e33 # g

    # Read in the appropriate .txt file (closest mass Bergeron table).
    
    file_to_read_in = full_bergeron_table_mass_filename(Mass) # string

    # Tabulate the results (this dataframe will be used below).
    
    dfr = closest_bergeron_table_mass_dataframe(Mass)

    # Extract the closest mass for the fiducial model. 
    
    with open(file_to_read_in, 'r') as f:

        lines = f.readlines()

        closest_mass = lines[0].strip().split()[1]

        mmsun = float(closest_mass) # decimal, now stored in memory

        m_berg = mmsun*Msun

    # Parameters of interest for the interpolation are below (converted to arrays): 

    t_berg = dfr['Teff'].astype(float)
    t_berg = t_berg.to_numpy()
    
    g_berg = 10**dfr['log g'].astype(float)
    g_berg = g_berg.to_numpy()

    r_berg = np.sqrt(G*m_berg/g_berg).astype(float) # already an array 

    # Interpolated radius: this represents the "reference" radius of the star, through 
    # interpolation of the Bergeron data, at the WDEC model teff:
    
    r_berg_interpolator_fcn = interpolator_function(t_berg, r_berg) 
    r_interpolated = r_berg_interpolator_fcn(Teff) # single number

    # Case 1: Return the interpolated radius.
    
    if which_filter==None: 

        return r_interpolated # single number
    
    # Case 2: Return the interpolated absolute magnitude. 

    elif which_filter is not None:

        # Proceed with absolute magnitudes:
        
        mag_berg = dfr[f"{which_filter}"].astype(float)
        mag_berg = mag_berg.to_numpy()

        mag_berg_interpolator_fcn = interpolator_function(t_berg, mag_berg) 

        # Interpolated absolute magnitude: this represents the "reference" absolute magnitude 
        # of the star, through interpolation of the Bergeron data, at the WDEC model teff. 
        
        mag_interpolated = mag_berg_interpolator_fcn(Teff) # single number

        # Finally, compute the (predicted) absolute magnitude of the model.
        
        results_tag_fname = other_functions.unique_results_dat_file_id(Teff, Mass, 
                            Menv, Mhe, Mh, 
                            He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env, 
                            diff_coeff_He_base_pure_He,
                            alpha, h1x100, h2x100, h3x100,
                            w1x100, w2x100, w3x100, w4x100)
        
        wdec_model = other_functions.specific_model(results_tag_fname)
        
        r_wdec = wdec_model.get_radius() # actual (physical) radius for the WDEC model.
        
        scale_factor = (r_wdec/r_interpolated)**2 
        
        computed_absolute_mag = -2.5*np.log(scale_factor) + mag_interpolated 

        return computed_absolute_mag 