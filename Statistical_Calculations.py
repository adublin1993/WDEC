#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:35:42 2025

@author: adublin
"""

import numpy as np

# Import the relevant notebook(s)
import Tabulation_Plots_Saving_Files as other_functions
import Bergeron_Interpolation as berg

def calc_astrometric_quality_of_fit(sigma_mag, obs_mag, interp_mag):
    """
    This function calculates the weighted square difference term between the 
    interpolated/model absolute magnitude and the measured absolute magnitude. 

    Input:
    - absolute magnitude uncertainty (float)
    - observed absolute magnitude (float)
    - interpolated absolute magnitude (float)

    Output:
    - astrometric quality of fit (float)
    """
    return (1/sigma_mag**2)*(obs_mag-interp_mag)**2

def computed_Gaia_absolute_magnitude_and_error(phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error,
                                      parallax_in_mas, parallax_error_in_mas):
    """
    This function will compute Gaia absolute magnitude and its uncertainty. The required input
    quantities are available in the Gaia archive. The function will (likely) be updated in 
    the future to accept/return additional quantities, depending on what analysis is required. 

    Input:
    - phot_g_mean_mag: G-band mean (apparent) magnitude (float)
    - phot_g_mean_flux: G-band mean flux, in electron/sec (float)
    - phot_g_mean_flux_error: Error on G-band mean flux, in electron/sec (float)
    - parallax_in_mas: Gaia parallax, provided in milli-arcseconds (float)
    - parallax_error_in_mas: Standard error on parallax, provided in milli-arcseconds (float)

    Output: 
    - G-band absolute magnitude (float)
    - Error on G-band absolute magnitude (float)
    """

    ### Compute the error on the apparent magnitude, where m(F) = -2.5log10(F) + constant

    # Error calculation: 
    # Sigma = |dm/dF| * sigma_F = (2.5 * 1/(F*log10)) * sigma_F = 2.5 * (sigma_F/F)
    
    sigma_app_mag = 2.5 * (phot_g_mean_flux_error/phot_g_mean_flux)
    
    ### Compute the absolute magnitude. 
    
    parallax_arcsec = (1/1000) * parallax_in_mas # parallax in arcseconds
    sigma_parallax_arcsec = (1/1000) * parallax_error_in_mas # parallax error in arcseconds

    # Error on distance can be obtained from error on parallax:
    
    d_pc = 1/parallax_arcsec # distance in pc
    
    # Error calculation: 
    # Sigma = |d(distance)/d(parallax)| * sigma_parallax = (1/parallax)^2 * sigma_parallax
    
    sigma_d_pc = (1/parallax_arcsec**2) * sigma_parallax_arcsec # error on distance in pc
    
    # The distance modulus (mu) is given by: mu = m - M = 5*log10(d) - 5, where d is in pc.
    
    dist_modulus = 5*np.log10(d_pc)-5
    
    # Error calculation:
    # Sigma = |d(mu)/d(distance)| * sigma_distance = (5 * 1/(d*log10)) * sigma_d = 5 * (sigma_d/d)
    
    sigma_mu = 5*(sigma_d_pc/d_pc)

    ### Compute the absolute magnitude and error on the absolute magnitude.

    # By propagation of errors, the error on the absolute magnitude involves a sum of two 
    # terms: the error on the apparent magnitude and the error on the distance modulus. This 
    # follows from the fact that M is a function of two variables: M(m, mu) = m - mu.
    
    abs_mag = phot_g_mean_mag - dist_modulus 
    
    sigma_abs_mag = np.sqrt(sigma_app_mag**2 + sigma_mu**2)
    
    return abs_mag, sigma_abs_mag
    
# Calculate the quality function S (modified from older code). 

def calc_s(measured_period_array, model_period_array, 
           measured_period_uncertainty_array, 
           Bergeron_filter_if_applicable, num_dof = None,
           include_absolute_magnitude = False,
           measured_absolute_magnitude = None,
           measured_absolute_magnitude_uncertainty = None,
           include_detailed_error_messages = False,
           return_sigma_rms = False,
           Teff=None, Mass=None, Menv=None, Mhe=None, Mh=None, 
           He_abund_mixed_CHeH_region=None, diff_coeff_He_base_env=None, 
           diff_coeff_He_base_pure_He=None, alpha=None, h1x100=None, 
           h2x100=None, h3x100=None, w1x100=None, w2x100=None, w3x100=None,
           w4x100=None):
    """
    This function is essential. It calculates the quality function (chi- 
    squared) value given a set of model periods and observed periods, with
    their uncertainties. When applicable, it is written flexibly to be able 
    to accommodate absolute magnitude measurements, so that the quality of 
    fit can be calculated either with or without astrometric considerations. 
    This is the function that is the bedrock of the Nelder Mead minimization 
    routine. This function can also calculate the sigma_rms value (see note
    below). 

    IMPORTANT: If the chi-squared calculation incorporates absolute magnitude,
    the user *must* specify the fiducial model parameters (default None). If
    these parameters are not specified, the code will *not* work. It is not
    necessary to provide these parameters if absolute magnitude is not meant to 
    be incorporated into the chi-squared calculation (ultimately the parameters
    would be ignored in this circumstance). 

    IMPORTANT: If the analysis is to exclude the consideration of absolute 
    magnitude, the "Bergeron_filter_if_applicable" parameter should be set 
    (explicitly) to None. 

    Note: This function was originally written with many (arguably excessive)
    detailed error messages. As the code was revised to handle (and compute) 
    much more computationally expensive models, I decided to make these messages 
    "unexecuted" by default. The original intention was to provide user-friendly 
    error messages handling many different scenarios, since this is such a 
    critical function. However, this is not feasible or computationally wise 
    for higher-dimensional calculations. This feature can be "turned on", of 
    course, but this is left up to user discretion and will be based on user 
    needs.

    Note: The number of degrees of freedom (dof) should be interpreted as the 
    number of varied (free) model parameters. The default here is None; in future 
    calls to this function (e.g., for actual trial runs), this parameter can be 
    updated, as necessary. 

    Note: By default, only the reduced chi-squared or chi-squared value is 
    returned. However, if the user wishes to also return the sigma_rms value,
    the "return_sigma_rms" boolean can be set to True. In this case, both the
    (reduced) chi-squared and sigma_rms values are returned, and the output 
    would be a tuple. 
    
    Input: 
    - measured periods (array only)
    - all model periods (array only)
    - measured period uncertainties (array only)
    
    Optional input:
    - measured absolute magnitude (float or int)
        *** IMPORTANT: If the analysis is to incorporate absolute magnitude, then
        all WDEC model input parameters (default None) *must* be provided for the 
        function to run. 
    - measured absolute magnitude uncertainty (float)
    - number of degrees of freedom (default None; otherwise int)
    
    Output: 
    - Chi-squared or reduced chi-squared (float)

    Optional Output:
    - sigma_rms (float); see above, returned within a tuple
    """

    if len(model_period_array) == 0:

        ### OLD APPROACH (not helpful when this issue actually arises): Don't bother with 
        # anything else below (i.e., don't both calculating chi-squared). 
            
        # warnings.warn("Array of model periods is empty. The quality function cannot be calculated. 
                        # Returning NaN for chi-squared (no calculation performed).")
        
        # return np.nan

        ### NEW APPROACH (actually addresses the issue): Compute a model on the fly, save the 
        # results, and redefine the model array. 

        # Compute the model
        other_functions.compute_wdec_model(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
                        diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, 
                        h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, w4x100)
        
        # Save all output files (good to have all of them)
        other_functions.name_and_save_wdec_output_files(Teff, Mass, Menv, Mhe, Mh, 
                        He_abund_mixed_CHeH_region, diff_coeff_He_base_env, 
                        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, 
                        h3x100, w1x100, w2x100, w3x100, w4x100, need_calcperiods=True, 
                        need_check_dat=True, need_chirt_dat=True, need_corsico_dat=True, 
                        need_cpvtga1_dat=True, need_deld_dat=True, need_discr_dat=True, 
                        need_epsrt_dat=True, need_gridparameters=True, need_kaprt_dat=True, 
                        need_lrad_dat=True, need_modelp_dat=True, need_output_dat=True, 
                        need_pform_dat=True, need_prop_dat=True, need_radii_dat=True, 
                        need_reflection2_dat=True, need_results_dat=True, need_struc_dat=True, 
                        need_tape18_dat=True, need_tape19_dat=True, need_tape28_dat=True, 
                        need_tape29_dat=True, need_temprp_dat=True)

        # Read in the calcperiods file to redefine the model array.
        f_ext = other_functions.unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, 
                        He_abund_mixed_CHeH_region, diff_coeff_He_base_env, 
                        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, 
                        h3x100, w1x100, w2x100, w3x100, w4x100, name='calcperiods')

        model_period_array = other_functions.read_in_calcperiods(calcperiods_file_ext = f_ext)
        
    ##############################################################################################
    ################### SKIP OVER THIS PART OF THE CODE (BELOW). SEE DOCSTRING ###################
    ##############################################################################################
    
    if include_detailed_error_messages:
    
        # The user must provide an array of periods consistent with the  
        # array of measured period uncertainties. 
        
        if len(measured_period_array) != len(measured_period_uncertainty_array):
    
            raise ValueError("The array lengths of the observed (measured) periods and "
                             "their associated uncertainties do not match.")
        
        # If the user attempts to include absolute magnitudes into the analysis but does
        # not provide uncertainties or measurements (or both), raise error messages
        # accordingly.
        
        if include_absolute_magnitude:
                
            list_of_required_params = [
                'Teff', 'Mass', 'Menv', 'Mhe', 'Mh', 'He_abund_mixed_CHeH_region',
                'diff_coeff_He_base_env', 'diff_coeff_He_base_pure_He', 'alpha', 
                'h1x100', 'h2x100', 'h3x100', 'w1x100', 'w2x100', 'w3x100', 'w4x100'
            ]
            
            if not any(list_of_required_params):
    
                raise ValueError("No input model parameters have been provided. All parameters "
                "are required for analysis \n involving absolute magnitude.")
    
            # This line will raise an error if at least one of the input model parameters is 
            # (are) not provided:
            
            elif (list_of_required_params and len(list_of_required_params) < 16):
    
                missing_pars = [param for param in list_of_required_params if 
                                param not in list_of_required_params]
                
                if len(missing_pars) == 0:
                    
                    # This line is techinically unnecessary since some parameters are missing (by
                    # construction, see above). Included only for readability and completeness.
                    
                    pass
    
                else:
                    
                    if len(missing_pars) == 1: 
    
                        which_missing = f"{missing_pars[0]}"
                    
                    elif len(missing_pars) > 1: 
    
                        which_missing = ""
                        which_missing = '; '.join(missing_pars)
    
                    raise ValueError(f"Missing fiducial model parameter(s): {which_missing}")
    
            # No valid filter provided (below). Raise the appropriate error messages:
            
            elif Bergeron_filter_if_applicable is None:
                
                if measured_absolute_magnitude is None and measured_absolute_magnitude_uncertainty is None:
                    
                    raise TypeError("No bandpass filter provided. No measured absolute magnitude provided. "
                                   "No measured absolute magnitude uncertainty provided.")
                    
                elif measured_absolute_magnitude is None:
                    
                    raise TypeError("No bandpass filter provided. No measured absolute magnitude provided.")
                    
                elif measured_absolute_magnitude_uncertainty is None:
                    
                    raise TypeError("No bandpass filter provided. No measured absolute magnitude uncertainty "
                                   "provided.")
    
                else:
                    
                    raise TypeError("No bandpass filter provided.")
    
            # A valid filter is provided. Raise similar error messages:
            
            else:
    
                if measured_absolute_magnitude is None or measured_absolute_magnitude_uncertainty is None:
    
                    if measured_absolute_magnitude is None and measured_absolute_magnitude_uncertainty is None:
                        raise TypeError("Neither the measured absolute magnitude nor the corresponding uncertainty "
                        "has been \n provided.")
                    
                    elif measured_absolute_magnitude is None:
        
                        raise TypeError("No measured absolute magnitude provided.")
                        
                    elif measured_absolute_magnitude_uncertainty is None:
        
                        raise TypeError("No measured absolute magnitude uncertainty provided.")
                                    
            # This block ensures that the data structures are nothing but floats/ints or 
            # None type.
            
            if (
                (not isinstance(measured_absolute_magnitude, (float, int)) and \
                 measured_absolute_magnitude is not None) or \
                (not isinstance(measured_absolute_magnitude_uncertainty, (float, int)) and \
                 measured_absolute_magnitude_uncertainty is not None)
            ):
                
                raise TypeError("At least one of the astrometric inputs has an unexpected type " 
                                "(float or int).")
                
            # Similar error message:
            
            if measured_period_uncertainty_array is None or measured_period_array is None:
    
                missing = ""
    
                if measured_period_uncertainty_array is None and measured_period_array is None:
         
                    missing += "measured period array; measured period uncertainty array"
                
                elif measured_period_uncertainty_array is None and measured_period_array is not None:
                        
                    missing += "measured period uncertainty array"
    
                elif measured_period_uncertainty_array is not None and measured_period_array is None:
    
                    missing += "measured period array"
                    
                raise TypeError(f"The following parameters are required but have not been provided: "
                f"{missing}.")
    
            else: 
                
                # Define both weight functions, only after exhausting all other error messages.
                # This will be more computationally efficient (in terms of memory allocation). 
                
                w_obs_period = 1/(measured_period_uncertainty_array**2) # array
    
        # This "elif" statement was used to make the code more readable; it can be recast as an "else"
        # statement.
    
        elif include_absolute_magnitude == False:
    
            print(
                "The code was instructed to calculate the quality function without "
                "any absolute magnitude quantities. Only the array of measured periods, "
                "the array of fiducial model periods, and the array of measured period "
                "uncertainties are used in the chi-squared calculation. All other inputs "
                "are irrelevant and have likewise been ignored."
            )
    
            # Define the (single) weight function.
            w_obs_period = 1/(measured_period_uncertainty_array**2) # array 

    ##############################################################################################
        ################### SKIP RIGHT TO THIS PART OF THE CODE (BELOW) ###################
    ##############################################################################################
    
    else: 
        
    # The following code holds for both cases (i.e., including or excluding absolute magnitude
    # considerations):

        # Define the (first) weight function outside the for-loop. This will be more computationally 
        # efficient (in terms of memory allocation): 

        # Weight function (measured periods), used in either scenario
        w_obs_period = 1/(measured_period_uncertainty_array**2) # array 

        # Instantiate the relevant arrays
        S_sq_per_arr = np.zeros(len(measured_period_array))

        if return_sigma_rms:

            # This array will only be needed when calculating sigma_rms
            all_per_sq_diffs_arr = np.zeros(len(measured_period_array))
    
        # Regardless: calculate the quality function for each measured period
        for i in range(len(measured_period_array)):
    
            # Consider only one measured period at a time
            particular_measured_period = measured_period_array[i]
    
            # Pointwise absolute difference of the measured period with the entire 
            # set of model periods
            
            diff = abs(particular_measured_period - model_period_array)
            
            # Find the index of the minimum absolute difference
            index_closest_model_period = np.argmin(diff)
    
            # Use this index to identify the corresponding model period
            closest_model_period = model_period_array[index_closest_model_period]

            # Weighted square difference for each of the periods
            # (This is needed regardless of whether absolute magnitude is included 
            # in the analysis or not.)
            S_sq_i = w_obs_period[i]*(particular_measured_period - closest_model_period)**2

            # Assign the value to the original array
            S_sq_per_arr[i] = S_sq_i

            if return_sigma_rms:

                # Square differences of periods (used below)
                per_sq_diff = (particular_measured_period - closest_model_period)**2

                # Assign all square (period) differences to the new array, needed regardless
                all_per_sq_diffs_arr[i] = per_sq_diff

        # Outside the loop: Calculate the sum of all weighted square (period) differences
        sum_of_all_weighted_per_sq_diffs = np.sum(S_sq_per_arr)
                
        # Outside the loop: Calculate the sum of all non-weighted square (period) differences
        if return_sigma_rms:

            # Needed (only) for sigma_rms
            sum_of_all_per_sq_diffs = np.sum(all_per_sq_diffs_arr)

        # Calculate the (optional) additional term for the chi-squared sum 
        if include_absolute_magnitude:

            interpolated_mag = berg.bergeron_interpolation(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
                                                     diff_coeff_He_base_env, diff_coeff_He_base_pure_He, 
                                                     alpha, h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, 
                                                     w4x100, which_filter = Bergeron_filter_if_applicable)

            # Needed for chi-squared and reduced chi-squared
            weighted_sq_diff_mag_term = calc_astrometric_quality_of_fit(measured_absolute_magnitude_uncertainty,
                                                            measured_absolute_magnitude, interpolated_mag)
            # Chi-squared with absolute magnitude
            chi_sq_total_sum_with_mag = sum_of_all_weighted_per_sq_diffs + weighted_sq_diff_mag_term 

        # Final step: Return (reduced) chi-squared and/or sigma_rms, either with or without 
        # absolute magnitude incorporated into the calculation:
        
        # With absolute magnitude
        if include_absolute_magnitude:
                
            if return_sigma_rms:

                if num_dof is None or num_dof == 0:

                    # (Ordinary) chi-squared & sigma_rms
                    return ((1/(len(measured_period_array)+1)) * chi_sq_total_sum_with_mag, \
                            np.sqrt((1/len(measured_period_array)) * \
                                    sum_of_all_per_sq_diffs))
        
                elif num_dof > 0:

                    # (Reduced) chi-squared & sigma_rms
                    return ((1/((len(measured_period_array)+1)-num_dof)) * chi_sq_total_sum_with_mag, \
                            np.sqrt((1/len(measured_period_array)) * \
                                    sum_of_all_per_sq_diffs)) 
 
            elif not return_sigma_rms:

                if num_dof is None or num_dof == 0:

                    return (1/(len(measured_period_array)+1)) * chi_sq_total_sum_with_mag

                elif num_dof > 0:

                    return (1/((len(measured_period_array)+1)-num_dof)) * chi_sq_total_sum_with_mag

        # Without absolute magnitude
        elif not include_absolute_magnitude:

            if return_sigma_rms:

                if num_dof is None or num_dof == 0:

                    # (Ordinary) chi-squared & sigma_rms
                    return ((1/len(measured_period_array)) * sum_of_all_weighted_per_sq_diffs, \
                        np.sqrt((1/len(measured_period_array)) * sum_of_all_per_sq_diffs))

                elif num_dof > 0:

                    # (Reduced) chi-squared & sigma_rms 
                    return ((1/(len(measured_period_array)-num_dof)) * sum_of_all_weighted_per_sq_diffs, \
                        np.sqrt((1/len(measured_period_array)) * sum_of_all_per_sq_diffs)) 
                        
            elif not return_sigma_rms:

                if num_dof is None or num_dof == 0:

                    # (Ordinary) chi-squared 
                    return (1/len(measured_period_array)) * sum_of_all_weighted_per_sq_diffs

                elif num_dof > 0:

                    # (Reduced) chi-squared
                    return (1/(len(measured_period_array)-num_dof)) * sum_of_all_weighted_per_sq_diffs
      
