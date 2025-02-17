#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:51:20 2025

@author: adublin
"""

import os
import numpy as np
import pandas as pd

# Import the other relevant notebook(s):
import Tabulation_Plots_Saving_Files as other_functions
import Statistical_Calculations as stats
import Bergeron_Interpolation as berg
    
class mode_comparison:
    """
    This class is designed to access information that can be helpful in mode comparison.
    One might use this to access the nearest model period (mode) for a given observed period.

    Note that this code features similar code from previous functions, so it is not
    entirely "new", and a lot of the functionality can be achieved through other functions.
    However, it is computationally efficient to be able to access these results and avoid 
    computationally expensive steps (e.g., reading in an entire Pandas dataframe when only 
    a subset of the data is needed). 
    
    """
    def __init__(self, Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
        diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, 
        h3x100, w1x100, w2x100, w3x100, w4x100):

        # Model parameters:
        self.Teff = Teff
        self.Mass = Mass
        self.Menv = Menv
        self.Mhe = Mhe
        self.Mh = Mh
        self.He_abund_mixed_CHeH_region = He_abund_mixed_CHeH_region
        self.diff_coeff_He_base_env = diff_coeff_He_base_env
        self.diff_coeff_He_base_pure_He = diff_coeff_He_base_pure_He
        self.alpha = alpha 
        self.h1x100 = h1x100
        self.h2x100 = h2x100
        self.h3x100 = h3x100
        self.w1x100 = w1x100 
        self.w2x100 = w2x100 
        self.w3x100 = w3x100 
        self.w4x100 = w4x100 
        
    def closest_model_period(self, Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                alpha, h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, 
                w4x100, observed_period):

        # This function is intended to read in the results.dat file of the fiducial model 
        # to identify the closest (model) period to the observed period.

        self.observed_period = observed_period
        
        path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
        
        gridparams = [Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                      diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                      alpha, h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, 
                      w4x100]

        gridparams_rounded_decimals = [round(var, 2) if isinstance(var, float) \
                                       else var for var in gridparams]

        file_name_identifier = "_".join(map(str,gridparams_rounded_decimals))

        results_dat_full_path = path + 'results.dat_' + file_name_identifier

        # Compute and save the model if it doesn't already exist (to play it safe)
        
        if not os.path.isfile(results_dat_full_path):
            
            other_functions.compute_wdec_model(*gridparams)
            other_functions.name_and_save_wdec_output_files(*gridparams)

        # Read in the appropriate results.dat (.txt) file:

        with open(results_dat_full_path, 'r') as file:

            lines = file.readlines()

            # In some bizarre cases, there might be a specific line that should be skipped over.
            # The actual numerical data starts at line 9 (for a "normal" file). 

            model_pers = []

            for line in lines[9:]:

                if 'NOTE:' in line:
                    
                    continue

                period = float(line.strip().split()[2])

                model_pers.append(period)

            model_pers = np.array(model_pers)

            # Another safety measure to make sure there is actual data:
            
            if len(model_pers) == 0:

                other_functions.compute_wdec_model(*gridparams)
                other_functions.name_and_save_wdec_output_files(*gridparams)
            
            index_closest_model_period = np.argmin(np.abs(observed_period-model_pers))

            return model_pers[index_closest_model_period]

    def mode_of_closest_model_period(self, Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                alpha, h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, 
                w4x100, observed_period, k_only=False, ell_only=False):

        # This function is intended to read in the results.dat file of the fiducial model 
        # to extract the mode of the closest model period. The code is identical to that
        # above, with some minor changes. 

        self.observed_period = observed_period
        
        path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
        
        gridparams = [Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                      diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                      alpha, h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, 
                      w4x100]

        gridparams_rounded_decimals = [round(var, 2) if isinstance(var, float) \
                                       else var for var in gridparams]

        file_name_identifier = "_".join(map(str,gridparams_rounded_decimals))

        results_dat_full_path = path + 'results.dat_' + file_name_identifier

        # Compute and save the model if it doesn't already exist (to play it safe)
        
        if not os.path.isfile(results_dat_full_path):
            
            other_functions.compute_wdec_model(*gridparams)
            other_functions.name_and_save_wdec_output_files(*gridparams)

        # Read in the appropriate results.dat (.txt) file:

        with open(results_dat_full_path, 'r') as file:

            lines = file.readlines()

            # In some bizarre cases, there might be a specific line that should be skipped over.
            # The actual numerical data starts at line 9 (for a "normal" file). 

            ell_vals = []
            k_vals = []
            per_vals = []

            for line in lines[9:]:

                if 'NOTE:' in line:
                    
                    continue

                ell = int(line.strip().split()[0])
                k = int(line.strip().split()[1])
                period = float(line.strip().split()[2])
                
                ell_vals.append(ell)
                k_vals.append(k)
                per_vals.append(period)

            # List of tuples
            all_mode_ids = list(zip(k_vals, ell_vals))

        # First, another safety measure to make sure there is actual data:
        
        if len(per_vals) == 0:

            other_functions.compute_wdec_model(*gridparams)
            other_functions.name_and_save_wdec_output_files(*gridparams)
            
        # Now that all the periods have been collected, identify the tuple containing the mode 
        # (k,l) which corresponds to the closest model period. 

        index_closest_model_period = np.argmin(np.abs(observed_period-np.array(per_vals)))

        # Optional return flexibility of either k or ell, depending on user need: 
        
        if k_only and not ell_only:

            return all_mode_ids[index_closest_model_period][0]

        elif ell_only and not k_only:

            return all_mode_ids[index_closest_model_period][1]

        else:
            
            return all_mode_ids[index_closest_model_period]

def prelim_analysis(model_params, obs_pers, model_pers, obs_period_indices, 
                obs_errors, abs_mag_error, meas_abs_mag, num_dof, 
                Bergeron_filter_if_applicable, include_absolute_magnitude,
                already_provided_observed_periods, include_detailed_error_messages = False, 
                return_sigma_rms = False): 
    """ 
    This function returns information regarding the model parameters and related
    observational data. 
    
    Input:
    - True (model) parameters (int/float)
    - Observed periods, indices only (list of ints)
    - Observational errors (list or array)
    - Absolute magnitude error (float)
    - Measured absolute magnitude (float)
    - Degrees of freedom (int)
    - Do you want to include absolute magnitude in the calculations? (bool)
    - Did you input/provide an explicit list of observed periods? (bool)
    - Do you want very detailed error messages? (bool)
    - Do you want to return the sigma rms value? (bool)
    - Optional: Absolute magnitude filter (str)
    
    Output: Dictionary
    - True (model) parameters (list of int/float)
    - True observed periods (array, int/float)
    - Observational errors (array, int/float)
    - Observed periods (int/float)
    - Interpolated G-band magnitude (int/float)
    - Measured G-band magnitude (int/float)
    - Measured G-band magnitude error (int/float)
    - Astrometric weighted square difference (float)
    - Model r/r* (float)
    - Model l/l* (float)
    - Closest model period for each observed period (float)
    - Mode identifications (int)
    - Chi-squared values (float)
    """
    
    # Compute & save WDEC model (only if the file does not exist already)
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    needed_file_ext = other_functions.unique_specific_file_id(name='calcperiods', *model_params)
    full_path = os.path.join(path, needed_file_ext)

    if not os.path.isfile(full_path):
        
        other_functions.compute_wdec_model(*model_params)
        other_functions.name_and_save_wdec_output_files(*model_params)

    # This if-else block was incorporated to provide flexibility, especially for AD's first
    # analysis. If the user provides a list of observed periods, essentially skip this step.
    # Otherwise, add noise to the true periods provided.
    
    if already_provided_observed_periods:

        observed_periods_with_noise = obs_pers

    else:

        # This case is specifically included for AD's analysis. See KIC notebook for more info
        # (in which the "true periods" are treated as the "observed periods"). 
        
        observed_periods_with_noise = obs_pers + np.random.randn(len(obs_pers))*obs_errors

    # Create the final (equivalent) dictionary for the data. Have it include the closest period, for
    # the provided observed period. 
    
    all_closest_pers = []
    all_closest_full_modes = []
    all_closest_k_vals = []
    all_closest_ell_vals = []
    
    for i in range(len(obs_pers)):

        trial_model = mode_comparison(*model_params)
        
        closest_per = trial_model.closest_model_period(*model_params, obs_pers[i])
        closest_mode = trial_model.mode_of_closest_model_period(*model_params, obs_pers[i])

        all_closest_pers.append(closest_per)
        all_closest_full_modes.append(closest_mode)

        closest_k = closest_mode[0]
        closest_ell = closest_mode[1]
        
        all_closest_k_vals.append(closest_k)
        all_closest_ell_vals.append(closest_ell)

    # Convert to the appropriate data structures before executing the call
    # to calc_s. 

    try: 
    
        # Note that obs_pers here accounts for both cases (above). 
        observed_periods_with_noise = np.array(observed_periods_with_noise) 
        model_pers = np.array(model_pers)
        obs_errors = np.array(obs_errors)
            
        chi_sq_with_mag = stats.calc_s(observed_periods_with_noise, model_pers, obs_errors, Bergeron_filter_if_applicable,
                                 num_dof, True, meas_abs_mag, abs_mag_error, include_detailed_error_messages, 
                                 False, *model_params)
    
        chi_sq_no_mag = stats.calc_s(observed_periods_with_noise, model_pers, obs_errors, Bergeron_filter_if_applicable,
                                num_dof, False, meas_abs_mag, abs_mag_error, include_detailed_error_messages, 
                               False, *model_params)

        # Initialize the dictionary that will be used to display the results
        dict_results = {}
    
        # Model parameters:
        all_param_names = ['Teff', 'Mass', 'Menv', 'Mhe', 'Mh', 
              'He_abund_mixed_CHeH_region', 
              'diff_coeff_He_base_env', 
              'diff_coeff_He_base_pure_He', 
              'alpha', 'h1x100', 'h2x100', 'h3x100', 
              'w1x100', 'w2x100', 'w3x100', 'w4x100']
        
        for par_name, param_val in zip(all_param_names, model_params):
            dict_results[f"{par_name}"] = param_val
    
        # Underlying (model) periods (for each observed period):
        for i, measured_per in enumerate(observed_periods_with_noise):
            dict_results[f"Underlying Period $P_{i+1}$ (no noise)"] = model_pers[i]
    
        # Measured periods:
        # NB: Don't use the variable "observed_periods_with_noise" on the RHS; use "obs_pers", or else
        # you will see the noise reflected in the data (i.e., the columns should contain the same number).
        for i, measured_per in enumerate(observed_periods_with_noise):
            dict_results[f"Measured Period $P_{i+1}$ (includes noise)"] = obs_pers[i]
            
        # Closest model periods (for each observed period)
        for i, measured_per in enumerate(observed_periods_with_noise):
            dict_results[f"Closest Model Period $P_{i+1}$ (no noise)"] = all_closest_pers[i]
    
        # Error bars on each observed period (usually the same):
        for i, sigma in enumerate(obs_errors):
            dict_results[f"Error on Observed Period $P_{i+1}$"] = sigma
    
        # Interpolated absolute magnitude: need to call the appropriate class for the specific model
        this_iter_interp_mag = berg.bergeron_interpolation(which_filter = Bergeron_filter_if_applicable,
                                                     *model_params)
        dict_results['Interpolated Absolute Magnitude'] = this_iter_interp_mag
        
        # Measured absolute magnitude:
        dict_results['Measured Absolute Magnitude'] = meas_abs_mag
    
        # Error bar on measured absolute magnitude
        dict_results['Measured Absolute Magnitude Error'] = abs_mag_error

        # Radius and luminosity of (current) model
        this_results_dat = other_functions.unique_results_dat_file_id(*model_params)
        this_iter_model_properties = other_functions.specific_model(this_results_dat)
        
        this_iter_radius = this_iter_model_properties.get_rrsun()
        this_iter_lum = this_iter_model_properties.get_llsun()
        
        dict_results['Radius (r/r*)'] = this_iter_radius
        dict_results['Luminosity (L/L*)'] = this_iter_lum
        
        # Square difference (absolute magnitude term)
        dict_results[f"$w(i)*(Gsim-Gint)^{2}$"] = stats.calc_astrometric_quality_of_fit(
            abs_mag_error, meas_abs_mag, this_iter_interp_mag)

        # Mode number k for closest period
        for i, k in enumerate(all_closest_k_vals):
            dict_results[f"$k$ of Closest Model Mode Period $P_{i+1}$"] = all_closest_k_vals[i]
    
        # Mode number l for closest period
        for i, ell in enumerate(all_closest_ell_vals):
            dict_results[f"$l$ of Closest Model Mode Period $P_{i+1}$"] = all_closest_ell_vals[i]
    
        # Chi-squared (including absolute magnitude)
        dict_results["Chi-Squared (with absolute magnitude)"] = chi_sq_with_mag
    
        # Chi-squared (without absolute magnitude)
        dict_results["Chi-Squared (without absolute magnitude)"] = chi_sq_no_mag
        
        dict_results = pd.DataFrame([dict_results])
        dict_results.index = dict_results.index + 1

        return dict_results 
        
    except Exception as e:

        print(f"{e}")
        
        pass

def vary_single_parameter(which_param, start, end, step_size, fixed_true_params_dict,
                        obs_pers, obs_period_indices, obs_errors, 
                        abs_mag_error, meas_abs_mag, Bergeron_filter_if_applicable, 
                        num_dof, include_absolute_magnitude, already_provided_observed_periods, 
                        all_param_names = ["Teff", "Mass", "Menv", "Mhe", "Mh",
                        "He_abund_mixed_CHeH_region", "diff_coeff_He_base_env", 
                        "diff_coeff_He_base_pure_He", "alpha", "h1x100", "h2x100", 
                        "h3x100", "w1x100", "w2x100", "w3x100", "w4x100"],
                         return_both_df_and_number_of_modes = False):
    """ 
    This function is intended to be used to create a dataframe in which
    the essential model parameters are stored iteratively. It is meant to
    be used for single-parameter variation, but it can be used in more 
    general contexts to address more complex scenarios.

    Input:
    - Parameter to be varied (str), e.g., 'Teff', Mass, etc.
    - Starting/minimum value (float)
    - End/maximum value (float)
    - Dictionary of *fixed* model parameters (dict)
    - Observed periods (list)
    - Errors on observed periods (list)
    - Absolute magnitude error (float)
    - Measured absolute magnitude (float)
    - Degrees of freedom (int)
    - Do you want to include absolute magnitude in the calculations? (bool)
    - Did you input/provide an explicit list of observed periods? (bool)
    - Would you like both the dataframe (df) as well as each number (int) of 
      l=1 and l=2 modes?
    - Optional: indices corresponding to an explicit list of observed periods (list)
    - Optional: absolute magnitude filter (str), e.g., 'G3', 'R', etc.

    Output:
    - Dataframe (df)
    - Optional: number of l=1 and l=2 modes (int)
    """
    param_range = np.arange(start, end, step_size)
    all_results = []

    # Loop over the varied parameter range, while keeping all other parameters fixed
    # to the "true" values. Update the dictionaries as appropriate within the loop.
    
    for val in param_range:
        
        # Initialize a dictionary
        specific_set_of_true_params = {key: 0 for key in all_param_names}
        
        # Update the dictionary to reflect the actual fiducial parameters (both the
        # varied and fixed values)
        specific_set_of_true_params.update(fixed_true_params_dict)
        specific_set_of_true_params[which_param] = val
        
        # Extract these values into a list which preserves order
        par_vals = [specific_set_of_true_params[param] for param in all_param_names]

        # Each model in the desired parameter range has a list of (true model) periods.
        # This will need to be read in each time, since the periods are unique to each
        # model.

        f_ext = other_functions.unique_specific_file_id(name = 'calcperiods', *par_vals)
        fid_model_pers = other_functions.read_in_calcperiods(calcperiods_file_ext = f_ext)
        
        # Call to the function that computes the results to be populated
        results_this_iter = prelim_analysis(par_vals, obs_pers, fid_model_pers, obs_period_indices, 
                                obs_errors, abs_mag_error, meas_abs_mag, num_dof, 
                                Bergeron_filter_if_applicable, include_absolute_magnitude,
                                include_detailed_error_messages=False, return_sigma_rms=False, 
                                already_provided_observed_periods=already_provided_observed_periods)
    
        all_results.append(results_this_iter)

        # Finally, store the (separate) dataframe containing the mode identifications and 
        # periods. We will concatenate this dataframe at the end.

        # Current iteration model parameters
        f_ext_results_dat = other_functions.unique_results_dat_file_id(*par_vals) 

        # This step creates the primary dataframe
        df_modes_and_periods = other_functions.tabulate_results_dat(
            file_ext = f_ext_results_dat, return_and_show_entire_dataframe = False,
            tabulate_periods_and_modes = True) 

        # If the user would like to access both the dataframe and the number of l=1 and l=2
        # modes, an optional return statement can be added here (which calls the appropriate
        # function):
        
        if return_both_df_and_number_of_modes: 

            num_ell1, num_ell2 = other_functions.count_how_many_modes(f_ext_results_dat)
            
        # Concatenate this dataframe to the tail end of the previous one
        all_results[-1] = pd.concat([all_results[-1], df_modes_and_periods], axis = 1)
            
    # Concatenate all dataframes accordingly
    df = pd.concat(all_results, ignore_index=True)
    df.index = df.index + 1

    # Optional return statement (see above)
    if return_both_df_and_number_of_modes: 
        
        return df, num_ell1, num_ell2
    
    else:

        return df