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
    It is intended to be used as part of a bilinear interpolation process, which is
    outlined in subsequent functions. 
    
    Input
        - model mass (float or int)

    Output
        - Pandas dataframes (df)
    """

    if Mass <= 0.20*1000:

        if Mass < 0.15*1000:
        
            print(f"Fiducial model mass of {Mass/1000}M is considerably less than "
                  f"the minimum mass of 0.2M available in the DBV Bergeron " 
                  f"Data Table collection.")
            print('The Bergeron table for 0.2M will be used, but beware'
                  'that this may introduce significant inaccuracies.')
        
        table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                            'Table_Mass_0.2')

        df_lower_bound = process_bergeron_file_one_mass(table_mass_file_name)
        df_upper_bound = df_lower_bound 

        return df_lower_bound, df_upper_bound
            
    elif 0.20*1000 < Mass < 0.30*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.2')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.3')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.30*1000 < Mass < 0.40*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.3')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.4')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.40*1000 < Mass < 0.50*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.4')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.5')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.50*1000 < Mass < 0.60*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.5')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.6')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.60*1000 < Mass < 0.70*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.6')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.7')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.70*1000 < Mass < 0.80*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.7')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.8')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.80*1000 < Mass < 0.90*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.8')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.9')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 0.90*1000 < Mass < 1.00*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.9')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.0')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 1.00*1000 < Mass < 1.10*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.0')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.1')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound
        
    elif 1.10*1000 < Mass < 1.20*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.1')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.2')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound

    elif 1.20*1000 < Mass < 1.30*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.2')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.3')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = process_bergeron_file_one_mass(upper_table_mass_file_name)

        return df_lower_bound, df_upper_bound
        
    elif 1.30*1000 < Mass < 1.40*1000:
        
        print('Approaching the Chandrasekhar limit. The data table for 1.3M will be ' \
              'used. Beware that this may introduce significant inaccuracies.')

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.3')

        df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
        df_upper_bound = df_lower_bound

        return df_lower_bound, df_upper_bound

    elif 1.40*1000 <= Mass:
        
        raise ValueError(f"Chandrasekhar limit reached. The fiducial model for {Mass/1000}M is "
                         f"unphysical.")

        return None

    else:

        extremes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] 

        for m in extremes:

            if Mass == 1000*m:

                lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        f"Table_Mass_{m}")

                df_lower_bound = process_bergeron_file_one_mass(lower_table_mass_file_name)
                df_upper_bound = df_lower_bound

                return df_lower_bound, df_upper_bound

def full_bergeron_table_mass_filename(Mass):
    """
    This function is very similar to the previous one, except that we just
    need the file names for the bilinear interpolation. It outputs the full
    file names (str) of the Bergeron models whose masses are closest to the 
    fiducial mass (i.e., the data tables of both the lower and upper bound 
    masses). This is necessary for the bilinear interpolation. 

    Input:
    - test model mass (float or int)
    
    Output:
    - full "...Table_Mass_(...)" file paths (str)
    """

    if Mass <= 0.20*1000:

        if Mass < 0.15*1000:
        
            print(f"Fiducial model mass of {Mass/1000}M is considerably less than "
                  f"the minimum mass of 0.2M available in the DBV Bergeron " 
                  f"Data Table collection.")
            print('The Bergeron table for 0.2M will be used, but beware '
                  'that this may introduce significant inaccuracies.')
        
        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                            'Table_Mass_0.2')
        upper_table_mass_file_name = lower_table_mass_file_name

    elif 0.20*1000 < Mass < 0.30*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.2')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.3')

    elif 0.30*1000 < Mass < 0.40*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.3')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.4')
    
    elif 0.40*1000 < Mass < 0.50*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.4')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.5')
    
    elif 0.50*1000 < Mass < 0.60*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.5')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.6')
    
    elif 0.60*1000 < Mass < 0.70*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.6')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.7')
    

    elif 0.70*1000 < Mass < 0.80*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.7')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.8')
    
    elif 0.80*1000 < Mass < 0.90*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.8')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.9')
    
    elif 0.90*1000 < Mass < 1.00*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_0.9')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.0')
    
    elif 1.00*1000 < Mass < 1.10*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.0')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.1')

    elif 1.10*1000 < Mass < 1.20*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.1')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.2')

    elif 1.20*1000 < Mass < 1.30*1000:

        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.2')
        upper_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.3')

    elif 1.30*1000 < Mass < 1.40*1000:

        print('Approaching the Chandrasekhar limit. The data table for 1.3M will be ' \
              'used. Beware that there may be significant inaccuracies as a result.')
        
        lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        'Table_Mass_1.3')
        upper_table_mass_file_name = lower_table_mass_file_name
        
    elif 1.40*1000 <= Mass:
        
        raise ValueError(f"Chandrasekhar limit reached. The fiducial model for {Mass/1000}M is "
                         f"unphysical.")

    else:

        extremes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] 

        for m in extremes:

            if Mass == 1000*m:

                lower_table_mass_file_name = os.path.join(global_bergeron_table_mass_file_dir, 
                                        f"Table_Mass_{m}")
                upper_table_mass_file_name = lower_table_mass_file_name
        
    # Return the appropriate file names:
    
    return lower_table_mass_file_name, upper_table_mass_file_name
      
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
    This function performs a bilinear interpolation. This function interpolates 
    over the Bergeron effective temperatures and masses to calculate radii and 
    absolute magnitudes in the specified bandpass. Note that the input parameters 
    correspond to those of the fiducial model. 

    Input: 
        - all fiducial model parameters (ints and/or floats)
        - filter (str)
             - NB: The filter should *NOT* be set to None. By default, the
             filter is set to None to provide input flexibility for the user. 
    
    Available filters (optional): 
    'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks', 'Y', 'J', 'H', 'K', 'W1', 'W2',     
    'W3', 'W4', 'S3.6', 'S4.5', 'S5.8', 'S8.0', 'u', 'g', 'r', 'i', 'z', 'g', 'r', 
    'i', 'z', 'y', 'G2', 'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'
            
    Output: 
        - Bergeron absolute magnitude rescaled to the WDEC radius (float)
    """
    
    # Constants: 
    
    G = 6.67259e-8 # cm3 g-1 s-2 
    Msun = 1.99e33 # g

    # Store in memory the names of the Bergeron files to be used.
    
    lower_ref_mass_file, upper_ref_mass_file = full_bergeron_table_mass_filename(Mass) # strings

    # Tabulate the results (these dataframes will be used below).
    
    lower_ref_mass_df, upper_ref_mass_df = closest_bergeron_table_mass_dataframe(Mass)

    # Calculate the lower and upper bounds for the first interpolation. This
    # requires reading in the appropriate files.

    with open(lower_ref_mass_file, 'r') as f_lower: 

        lines_lower = f_lower.readlines()
        closest_mass_lower = lines_lower[0].strip().split()[1]
        mmsun_lower = float(closest_mass_lower) 
        m_berg_lower = mmsun_lower*Msun # lower bound

    with open(upper_ref_mass_file, 'r') as f_upper: 

        lines_upper = f_upper.readlines()
        closest_mass_upper = lines_upper[0].strip().split()[1]
        mmsun_upper = float(closest_mass_upper) 
        m_berg_upper = mmsun_upper*Msun # upper bound

    # Lower bound on radius

    t_berg_lower = lower_ref_mass_df['Teff'].astype(float)
    t_berg_lower = t_berg_lower.to_numpy()

    g_berg_lower = 10**lower_ref_mass_df['log g'].astype(float)
    g_berg_lower = g_berg_lower.to_numpy()

    r_berg_lower = np.sqrt(G*m_berg_lower/g_berg_lower).astype(float) 
    
    # Upper bound on radius

    t_berg_upper = upper_ref_mass_df['Teff'].astype(float)
    t_berg_upper = t_berg_upper.to_numpy()

    g_berg_upper = 10**upper_ref_mass_df['log g'].astype(float)
    g_berg_upper = g_berg_upper.to_numpy()

    r_berg_upper = np.sqrt(G*m_berg_upper/g_berg_upper).astype(float) 

    # Interpolator functions for radius
    
    r_berg_interpolator_lower = interpolator_function(t_berg_lower, r_berg_lower) 
    r_berg_interpolator_upper = interpolator_function(t_berg_upper, r_berg_upper) 

    # First interpolation (radius)--intermediate step
    
    r_interpolated_lower = r_berg_interpolator_lower(Teff) # single number
    r_interpolated_upper = r_berg_interpolator_upper(Teff) # single number

    if which_filter is not None:

        # First interpolation (over the effective temperatures of the lower 
        # and upper mass tracks):
        
        mag_berg_lower = lower_ref_mass_df[f"{which_filter}"].astype(float)
        mag_berg_lower = mag_berg_lower.to_numpy()

        mag_berg_upper = upper_ref_mass_df[f"{which_filter}"].astype(float)
        mag_berg_upper = mag_berg_upper.to_numpy()

        mag_berg_interpolator_lower = interpolator_function(t_berg_lower, mag_berg_lower)
        mag_berg_interpolator_upper = interpolator_function(t_berg_upper, mag_berg_upper)

        # The interpolated absolute magnitude represents the "reference" absolute 
        # magnitude of the star, through interpolation of the Bergeron data, at the 
        # WDEC model teff. 
        
        mag_int_lower = mag_berg_interpolator_lower(Teff) # single number
        mag_int_upper = mag_berg_interpolator_upper(Teff) # single number

        # Second interpolation (over the masses of the lower and upper mass tracks)
        second_interpolation_mag = interp1d([1000*mmsun_lower, 1000*mmsun_upper], 
                                    [mag_int_lower, mag_int_upper])

        second_interpolation_rad = interp1d([1000*mmsun_lower, 1000*mmsun_upper], 
                                    [r_interpolated_lower, r_interpolated_upper])

        ### Final results of the bilinear interpolation ###
        mag_interpolated_final = second_interpolation_mag(Mass)
        r_interpolated_final = second_interpolation_rad(Mass)

        # Finally, compute the (predicted) absolute magnitude of the model.
        results_tag_fname = other_functions.unique_results_dat_file_id(
                            Teff, Mass, Menv, Mhe, Mh, 
                            He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env, 
                            diff_coeff_He_base_pure_He,
                            alpha, h1x100, h2x100, h3x100,
                            w1x100, w2x100, w3x100, w4x100)
        
        wdec_model = other_functions.specific_model(results_tag_fname)
        
        r_wdec = wdec_model.get_radius() # actual (physical) radius for the WDEC model.
        
        scale_factor = (r_wdec/r_interpolated_final)**2 

        # Bergeron magnitude rescaled to the WDEC radius
        
        berg_mag_rescaled = -2.5*np.log(scale_factor) + mag_interpolated_final

        return berg_mag_rescaled 
 
