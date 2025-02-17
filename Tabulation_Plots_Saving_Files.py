#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:51:50 2025

@author: adublin
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd

# Import the other relevant notebook(s):
import Statistical_Calculations as stats

# Global paths (update this as appropriate):
global_bergeron_radius_file = '/Users/adublin/Desktop/WDEC/Bergeron_DBV_grid/NEWradii.dat'
global_bergeron_table_mass_file_dir = '/Users/adublin/Desktop/WDEC/Bergeron_DBV_grid/Bergeron_Tables/'

# List of the names of all parameters (easy to reference later on)
all_param_names = ['Teff', 'Mass', 'Menv', 'Mhe', 'Mh', 
                  'He_abund_mixed_CHeH_region', 
                  'diff_coeff_He_base_env', 
                  'diff_coeff_He_base_pure_He', 
                  'alpha', 'h1x100', 'h2x100', 'h3x100', 
                  'w1x100', 'w2x100', 'w3x100', 'w4x100']

def format_names_properly(name):
    """
    This function formats the string names for WDEC parameters and other 
    quantities that appear in plots (e.g., fractional mass coordinate, etc). 

    Input:
    - parameter description (str)

    Output:
    - formatted parameter name (str)
    """
    
    if name == 'omega':
        
        name = r'Ω $(\mathrm{s}^{-1})$'
    
    elif name == 'Teff':
        
        name = r'$\mathrm{T}_{\mathrm{eff}}$ (K)'
    
    elif name == 'Mass':
        
        name = r'Mass $(\mathrm{M}_{\odot})$x1000'
    
    elif name == 'Menv':
        
        name = r'$\mathrm{M}_{\mathrm{env}} (\mathrm{M}_{\odot})$'
        
    elif name == 'Mhe':
        
        name = r'$\mathrm{M}_{\mathrm{He}} (\mathrm{M}_{\odot})$x1000'
        
    elif name == 'Mh':
        
        name = r'$\mathrm{M}_{\mathrm{H}}$ $(\mathrm{M}_{\odot})$'
    
    elif name == 'He_abund_mixed_CHeH_region':
        
        name = r'He abund. mixed C/He/H region'
    
    elif name == 'diff_coeff_He_base_env':
    
        name = r'diff. coeff. He base env.'
    
    elif name == 'diff_coeff_He_base_pure_He':
        
        name = r'diff. coeff. He base pure He'
        
    elif name == 'alpha':
        
        name = r'α'
    
    elif name == 'h1x100':
        
        name = r'$\mathrm{h_{1}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'
    
    elif name == 'h2x100':
        
        name = r'$\mathrm{h_{2}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'
    
    elif name == 'h3x100':
        
        name = r'$\mathrm{h_{3}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'
    
    elif name == 'w1x100':
        
        name = r'$\mathrm{w_{1}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'
    
    elif name == 'w2x100':
        
        name = r'$\mathrm{w_{2}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'
    
    elif name == 'w3x100':
        
        name = r'$\mathrm{w_{3}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'
    
    elif name == 'w4x100':
        
        name = r'$\mathrm{w_{4}}$x100 $(\mathrm{M}$/$\mathrm{M}_{\odot})$'

    # Non-WDEC parameter name formatting:

    elif name == 'partt': # Fractional mass coordinate on log scale (corsico.dat)

        name = r'$-\log(1-\mathrm{M}/\mathrm{M}_{\odot})$'

    elif name == 'ang_freq': # omega^2 (angular frequency, squared)

        name = r'$\omega^{2}$'

    elif name == 'bvfreq': # Brunt-Vaisala frequency 

        name = r'$N^{2}$ (Brunt-Vaisala Frequency)'

    elif name == 'acous': # Lamb frequency 

        name = r'$L^{2}$ (Lamb Frequency)'

    elif name == 'mmsun': # Fractional mass

        name = r'$\mathrm{M}_{\mathrm{r}}/\mathrm{M}_{\odot}$'

    elif name == 'Pk': # Period of mode with radial order k

        name = r'$\mathrm{P}_{k} \, (\mathrm{s})$'

    elif name == 'LogEkink': # Log Kinetic Energy for mode with radial order k

        name = r'$\mathrm{Log}(E_{\mathrm{kin}})_{k} \, \, (\mathrm{erg})$'

    elif name == 'ell1': # Ell=1 modes

        name = r'$\ell=1$'

    elif name == 'ell2': # Ell=2 modes

        name = r'$\ell=2$'

    elif name == 'ckl': # Ledoux coefficient 

        name = r'$c_{k\ell}$'

    elif name == 'rrsun': # Fractional radius 

        name = r'$\mathrm{r}/\mathrm{R}_{\odot}$'
    
    elif name == 'fr': # Fractional radius (not to be confused with rrsun)

        name = r'$\mathrm{r}/\mathrm{R}$'

    elif name == 'log_kappa': # Log opacity

        name = r'$\log(\kappa) \, \, (\mathrm{cm}^2/\mathrm{g})$'

    # Miscellaneous

    elif name == 'red_chi_sq': # Reduced chi-squared 

        name = r'$\chi_{\mathrm{red}}^{2}$'

    elif name == 'chi_sq': # Standard chi-squared 

        name = r'$\chi^{2}$'

    elif name == 'sigma_rms': # Sigma rms 

        name = r'$\sigma_{\mathrm{rms}}\,\mathrm{(s)}$'
        
    return name

def unique_results_dat_file_id(Teff, Mass, 
    Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
    diff_coeff_He_base_env, diff_coeff_He_base_pure_He, 
    alpha, h1x100, h2x100, h3x100, w1x100, w2x100, 
    w3x100, w4x100):
    """
    This function gives each results.dat file a unique identifier.
    This identifier is a string representing the file extension of 
    the results.dat file associated with the model whose parameters 
    are accepted as input.

    Input: 
    - model parameters (floats or ints)

    Output: 
    - results.dat file extension (str)
    """
    
    gridparams = [Teff, Mass, Menv, Mhe, Mh, 
                  He_abund_mixed_CHeH_region, 
                  diff_coeff_He_base_env, 
                  diff_coeff_He_base_pure_He, 
                  alpha, h1x100, h2x100, h3x100, 
                  w1x100, w2x100, w3x100, w4x100]

    gridparams_rounded_decimals = [round(var, 2) if isinstance(var, float) 
                else var for var in gridparams]

    file_name_identifier = "_".join(map(str,gridparams_rounded_decimals))
    
    file_ext = "results.dat_" + file_name_identifier
    
    return file_ext

def unique_specific_file_id(Teff, Mass, 
    Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
    diff_coeff_He_base_env, diff_coeff_He_base_pure_He, 
    alpha, h1x100, h2x100, h3x100, w1x100, w2x100, 
    w3x100, w4x100, name):
    """
    This function is exactly the same as above, but for any
    type of .dat file extension that might be desired (not just
    for results.dat). Note that, of course, this will also work 
    for results.dat. (I just wanted to create a function that will
    work in different contexts.)

    Input: 
    - model parameters (floats or ints)
    - specific file extension name (e.g., "corsico," "calcperiods,"
    "gridparameters", etc.)
    
    Output: 
    - .dat file extension (str)
    """
    
    gridparams = [Teff, Mass, Menv, Mhe, Mh, 
                  He_abund_mixed_CHeH_region, 
                  diff_coeff_He_base_env, 
                  diff_coeff_He_base_pure_He, 
                  alpha, h1x100, h2x100, h3x100, 
                  w1x100, w2x100, w3x100, w4x100]

    gridparams_rounded_decimals = [round(var, 2) if isinstance(var, float) 
                else var for var in gridparams]

    file_name_identifier = "_".join(map(str,gridparams_rounded_decimals))

    # The calcperiods file extension is different from the others (there is no .dat part)
    
    if name == 'calcperiods' or name == 'gridparameters':
        
        file_ext = f"{name}_" + file_name_identifier

    else: 
    
        file_ext = f"{name}.dat_" + file_name_identifier
    
    return file_ext

def parse_model_parameters(arbitrary_results_dat_file_ext):
    """
    This function starts with a results.dat file extension and 
    numerically parses the parameters. This can be helpful if there is
    a model that is supposed to exist but has not been computed and
    whose results (files) have not been saved. By having this function,
    we can add conditional checks within other functions to see if a 
    particular file exists. If it doesn't, the model can be created on 
    the fly. 

    Input:
        - results.dat file extension (str)
        
    Ouptut:
        - model parameters, each are floats or ints (tuple)
    """
    
    parsed_var_list = arbitrary_results_dat_file_ext.split("_")[1:]

    model_params = []

    # Ensure consistency with previous formatting of file names. Allow
    # the integers to appear without decimal points. Round the floats to
    # two decimal points. 
    
    try:
        
        for parsed_string in parsed_var_list:

            num = float(parsed_string) # first convert to floats
            
            if '.' in parsed_string:

                num = round(num, 2) # round to two decimal points
                
                model_params.append(num)

            else:

                if num.is_integer():
                    
                    num = round(num, 0) # no decimals for ints in file names
                    
                    num = int(num) # next convert to int, no decimals
                    
                    model_params.append(num)

        # Define each parameter manually:
        
        Teff = model_params[0]
        Mass = model_params[1] 
        Menv = model_params[2] 
        Mhe = model_params[3] 
        Mh = model_params[4]            
        He_abund_mixed_CHeH_region = model_params[5] 
        diff_coeff_He_base_env = model_params[6] 
        diff_coeff_He_base_pure_He = model_params[7] 
        alpha = model_params[8] 
        h1x100 = model_params[9] 
        h2x100 = model_params[10] 
        h3x100 = model_params[11] 
        w1x100 = model_params[12] 
        w2x100 = model_params[13] 
        w3x100 = model_params[14] 
        w4x100 = model_params[15]

    except ValueError as e:

        raise ValueError(f"Some numerical model parameter types may have been misinterpreted " \
                         f"during the results.dat file name parsing process. Error: {e}.")
    
    return (Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
            diff_coeff_He_base_env, diff_coeff_He_base_pure_He, 
            alpha, h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, 
            w4x100)

# Preliminary: Write a function that reads in calcperiods (l=1) and/or calcperiods
# (l=2). 

def read_in_calcperiods(calcperiods_file_ext, ell1_periods_only=False,
                       ell2_periods_only=False):
    """
    This function reads in the periods from the calcperiods output file. 
    It returns all periods as an array. The function provides flexibility
    for returning either the l=1 or l=2 periods (separately) as well.

    Input:
        - calcperiods file extension (str)

    Output:
        - default: both ell=1 and ell=2 periods (array)
        - optional: only ell=1 periods or ell=2 periods (array)
    """
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + str(calcperiods_file_ext)
    
    try: 
        
        calcperiods_headers = ['l', 'Period (s)']
        calcperiods_df = pd.read_csv(full_path,delimiter=r'\s+', skiprows=1, 
                        names=calcperiods_headers, dtype={'l':int})

        # Drop any rows with NaNs:
        calcperiods_df.dropna(how='any', inplace=True)

        if ell1_periods_only:

            ell1_pers = calcperiods_df.loc[calcperiods_df['l']==1, 'Period (s)']
            ell1_pers = np.array(ell1_pers)

            return ell1_pers

        elif ell2_periods_only:

            ell2_pers = calcperiods_df.loc[calcperiods_df['l']==2, 'Period (s)']
            ell2_pers = np.array(ell2_pers)

            return ell2_pers

        else:

            all_pers = calcperiods_df['Period (s)'].values
            all_pers = np.array(all_pers)
            
            return all_pers
        
    except FileNotFoundError:
        
        print(f"File not found: {full_path}")
        
def compute_wdec_model(Teff, Mass, Menv, Mhe, Mh, 
                       He_abund_mixed_CHeH_region, 
                       diff_coeff_He_base_env, 
                       diff_coeff_He_base_pure_He, 
                       alpha, h1x100, h2x100, h3x100, 
                       w1x100, w2x100, w3x100, w4x100
                      ):
    """
    This function computes a WDEC model and outputs the relevant files (including 
    the results.dat file). This step includes saving the files to the relevant 
    directories, deleting any superfluous files from older runs, etc. There is no
    actual "output" for this function (i.e., nothing is formally "returned").
    
    Input: 
        - model parameters (floats or ints)
    
    Output: 
        - NoneType
    """
    
    exe_path = '/Users/adublin/Desktop/WDEC/wdec-master'

    # Need to change directories (to the executable directory)
    # in order for this to work.
    
    os.chdir(exe_path)
    
    gridparams = [Teff, Mass, Menv, Mhe, Mh, 
                  He_abund_mixed_CHeH_region, 
                  diff_coeff_He_base_env, 
                  diff_coeff_He_base_pure_He, 
                  alpha, h1x100, h2x100, h3x100, 
                  w1x100, w2x100, w3x100, w4x100]
    
    gridparams_rounded_decimals = [round(var, 2) if isinstance(var, float) 
                else var for var in gridparams]
    
    formatted_gridparams = [f"{var:.2f}" if isinstance(var, float) else str(var) \
                            for var in gridparams_rounded_decimals]

    np.savetxt(os.path.join(exe_path, 'gridparameters'), formatted_gridparams, fmt='%s')
    
    os.system('./makedx_v20')
    
    # Switch back to current working directory (to play it safe).
    
    os.chdir(os.getcwd())
    
def name_and_save_wdec_output_files(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100, need_calcperiods=False, 
        need_check_dat=False, need_chirt_dat=False, need_corsico_dat=False,
        need_cpvtga1_dat=False, need_deld_dat=False, need_discr_dat=False, 
        need_epsrt_dat=False, need_gridparameters=False, need_kaprt_dat=False, 
        need_lrad_dat=False, need_modelp_dat=False, need_output_dat=False, 
        need_pform_dat=False, need_prop_dat=False, need_radii_dat=False,
        need_reflection2_dat=False, need_results_dat=False, need_struc_dat=False, 
        need_tape18_dat=False, need_tape19_dat=False, need_tape28_dat=False, 
        need_tape29_dat=False, need_temprp_dat=False,need_thorne_dat=False, 
        need_xhe_dat=False, need_xir_dat=False):
    """
    This function is independent of the compute_wdec_model function. However,
    it is intended to save all the files that are written after a model is 
    computed. It should be used in tandem with the compute_wdec_model function.
    There is an option to provide a list of the names of specific output files, 
    if necessary. By default, the return is NoneType. 

    IMPORTANT: if used for a single run, make sure you've computed the model 
    immediately before use. Otherwise, the file name might not correspond to 
    the output!
    
    Input:
        - model parameters (floats or ints)

    Output:
        - Default: NoneType
        - Optional: desired file names, as strings (list)
    """
    
    exe_path = '/Users/adublin/Desktop/WDEC/wdec-master'
    output_dir = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    
    # Here is a list of all the .dat WDEC output files after a run
    # is completed in the executable directory.
    
    calcperiods = 'calcperiods'
    check_dat = 'check.dat'
    chirt_dat = 'chirt.dat'
    corsico_dat = 'corsico.dat'
    cpvtga1_dat = 'cpvtga1.dat'     
    deld_dat = 'deld.dat'
    discr_dat = 'discr.dat'
    epsrt_dat = 'epsrt.dat'
    gridparameters = 'gridparameters'
    kaprt_dat = 'kaprt.dat'
    lrad_dat = 'lrad.dat'
    modelp_dat = 'modelp.dat'
    output_dat = 'output.dat'
    pform_dat = 'pform.dat'
    prop_dat = 'prop.dat'
    radii_dat = 'radii.dat'
    reflection2_dat = 'reflection2.dat'
    results_dat = 'results.dat'
    struc_dat = 'struc.dat'
    tape18_dat = 'tape18.dat'
    tape19_dat = 'tape19.dat'
    tape28_dat = 'tape28.dat'
    tape29_dat = 'tape29.dat'
    temprp_dat = 'temprp.dat'
    thorne_dat = 'thorne.dat'
    xhe_dat = 'xhe.dat'
    xir_dat = 'xir.dat'
    
    all_files = [calcperiods, check_dat, chirt_dat, corsico_dat,
                 cpvtga1_dat, deld_dat, discr_dat, epsrt_dat,
                 gridparameters, kaprt_dat, lrad_dat, modelp_dat,
                 output_dat, pform_dat, prop_dat, radii_dat,
                 reflection2_dat, results_dat, struc_dat, tape18_dat,
                 tape19_dat, tape28_dat, tape29_dat, temprp_dat,
                 thorne_dat, xhe_dat, xir_dat]
    
    gridparams = [Teff, Mass, Menv, Mhe, Mh, 
                  He_abund_mixed_CHeH_region, 
                  diff_coeff_He_base_env, 
                  diff_coeff_He_base_pure_He, 
                  alpha, h1x100, h2x100, h3x100, 
                  w1x100, w2x100, w3x100, w4x100]
    
    gridparams_rounded_decimals = [round(var, 2) if isinstance(var, float) 
                else var for var in gridparams]
    
    # Create a string to identify (and save) each file by its gridparameters.
    
    file_name_identifier = '_'.join(map(str,gridparams_rounded_decimals))

    for specific_file in all_files:

        copied_file = f"{specific_file}_{file_name_identifier}"
        shutil.copyfile(os.path.join(exe_path, specific_file), os.path.join(exe_path, copied_file))
        print("saved: ", copied_file)
        shutil.copyfile(os.path.join(exe_path, copied_file), os.path.join(output_dir, copied_file))
        os.remove(os.path.join(exe_path, copied_file))
        
    # This part of the code gives flexibility as to whether or not to
    # return the file name of a certain output file (or several):

    which_needed_files = [need_calcperiods, need_check_dat, need_chirt_dat, 
                need_corsico_dat, need_cpvtga1_dat, need_deld_dat, need_discr_dat, 
                need_epsrt_dat, need_gridparameters, need_kaprt_dat, need_lrad_dat, 
                need_modelp_dat, need_output_dat, need_pform_dat, need_prop_dat, 
                need_radii_dat, need_reflection2_dat, need_results_dat, need_struc_dat, 
                need_tape18_dat, need_tape19_dat, need_tape28_dat, need_tape29_dat, 
                need_temprp_dat, need_thorne_dat, need_xhe_dat, need_xir_dat]

    if any(which_needed_files): # check to see which ones are needed

        needed_file_paths = []
        
        for file_bool, file_name in zip(which_needed_files, all_files):

            if file_bool: # only collect those file names that evaluate to "model"

                full_needed_file_name = output_dir + f"{file_name}_{file_name_identifier}"

                needed_file_paths.append(full_needed_file_name)
        
        return needed_file_paths

    else:

        return None
    
def tabulate_results_dat(file_ext, return_llsun_rrsun_bolmag = False,
                         return_and_show_entire_dataframe = True,
                         calculate_fiducial_model_radius = False,
                         ell1_df_only=False, ell2_df_only=False,
                         tabulate_periods_and_modes = False):                      
    """
    Input: 
        - results.dat file extension (str)
        - three "needed" booleans (bool)
        - other "optional" booleans regarding dataframes (bool)
        
    Output: 
        - Default: Pandas dataframe of the results.dat file (df)
    
    NB: The user can modify one of the two optional booleans to 
        return either the l=1 or l=2 dataframes instead of the 
        entire dataframe.

    *** NB: If the user wants to return a table of all periods and 
    corresponding mode identifications, set the last boolean to
    True. Otherwise, this boolean can be ignored. 
    - This block will only be executed if (the boolean) 
    "return_and_show_entire_dataframe" is set to False. In this case, 
    the radius will not be returned (just the appropriate data table). 
    - It is important to note that occasionally there are duplicate 
    modes (k,l) with distinct periods. In this case, the code is written 
    such that only one of the periods is tabulated, as modes and periods 
    are expected to be 1-1 (for most plotting purposes). 

    Important note: It seems that for temperatures between 37000K
    and 38000K, the results.dat file contains an additional line
    (specifically at row/index 9), beginning with an ad hoc error
    message in the Fortran code ("NOTE: There are more than 100 
    period guesses in this scan.") This will prevent my code from
    properly executing, specifically, the "with open(...)" statement. 
    This shouldn't be an issue, because the typical temperature range
    for any meaningful analysis shouldn't exceed 30000K. Additional
    modifications to my code would be necessary to handle such cases,
    but these cases would typically not be physically realistic or
    meaningful.
    
    The following parameters can be output if specified with the
    boolean parameters:
    
        L/Lsun (float)
        R/Rsun (float)
        Bolometric magnitude (Mbol) of fiducial model (float)
        Radius (R) of fiducial model (float)

    Order of the (fullest) return statement:
        Pandas dataframe (results.dat), L/Lsun, R/Rsun, Mbol, R
    
    Default for the first three booleans: (F, T, T)
        Combinations:
        1)	Default (F, T, T): 1 output (dataframe)
            - if a boolean is provided, the l=1 or l=2 dataframe is
            returned instead.
        2)	(F, T, F): 1 output (dataframe)
            - if a boolean is provided, the l=1 or l=2 dataframe is
            returned instead.
        3)   (F, F, T): 1 output (fiducial model radius)
        4)   (F, F, F): omitted (non-sensical case)
        5)	(T, T, T): all 5 outputs
            - if a boolean is provided, the l=1 or l=2 dataframe is
            returned instead.
        6)   (T, T, F): dataframe, llsun, rrsun, bolometric magnitude 
            - if a boolean is provided, the l=1 or l=2 dataframe is
            returned instead.
        7)	(T, F, T): 4 outputs in the same order, just no dataframe
        8)   (T, F, F): 3 outputs (llsun, rrsun, bolometric magnitude)
    """
    
    # Relevant constants:

    Rsun = 6.957e10 # cm
    Mbol_sun = 4.74
    
    # Column names of the WDEC output file:
    
    results_headers = ['l', 'k', 'Period (s)', 'Int. Period (s)', \
                       'Log KE', 'y3', 'Nodes (y1)', 'Nodes (y2)', \
                       'plmode', 'flke', 'ckl'] 

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The results.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired results.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.

        try: 
            
            model_params = parse_model_parameters(file_ext)
    
            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full results.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_results_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list
    
            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):
    
                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:
    
            # There might be some unforeseen error in parsing parameters, etc. 
    
            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
        
        with open(full_path, 'r') as file:

            # WDEC contains erroneously numbered k values. This block of code
            # will ensure that the actual k values are correctly labeled. There 
            # is no reference to the current columns that contain the supposed
            # k values. The assumption is that the list of k values is strictly
            # increasing (e.g., 1,2,3,...). 

            current_ell = None
            current_k = 0
            modified_list_of_k = []
            periods = [] # used later on
            
            lines = file.readlines()

            # Make sure to properly label the k, ell, and period columns.
            
            for i, line in enumerate(lines[9:]):

                ell = line.strip().split()[0] # ells
                ell = int(float(ell))
                
                period = line.strip().split()[2] # periods
                period = float(period)
                periods.append(period) # used later on

                # Make sure that k resets once there is a change from l=1 to l=2.
                
                if ell != current_ell:
                    current_ell = ell
                    current_k = 1
                else:
                    current_k += 1

                modified_list_of_k.append((current_ell, current_k))
                
            modified_k_series = pd.Series(modified_list_of_k) 

            # This if-block might be necessary if there is a weird error message that pops 
            # up (this is rare, but it can happen):
            
            if 'NOTE:' in lines[9]:

                skip_rows = 10 # Can't skip 9 rows here, this needs to be set to 10 to work

            else:

                skip_rows = 8
                
            l_div_lsun = lines[2].strip().split()[11]
            l_div_lsun = float(l_div_lsun)
            
            Mbol_model = -2.5*np.log10(l_div_lsun)+Mbol_sun
            Mbol_model = round(Mbol_model, 2)
    
            r_div_rsun = lines[3].strip().split()[10]
            r_div_rsun = float(r_div_rsun)
            
            fiducial_model_radius = r_div_rsun * Rsun

        # File has been closed. Create the Pandas dataframe. Reference the correct 
        # number of rows to be skipped (as defined above). 
    
        results_df = pd.read_csv(full_path,delimiter=r'\s+', names=results_headers, \
                                 skiprows=skip_rows)

        # Additional modifications:

        try:
            
            results_df['k'] = modified_k_series.values # replace with the actual k values (see above)
            results_df.insert(2, 'm', 0)
            results_df.index = results_df.index + 1

        except Exception as e:

            raise Exception(f"Double-check the length of the modified list of k values. Ensure \n"
            f"that it matches the length of the overall dataframe: {e}.")

        # Case (T=default, T=default, T=default): 
        
        if return_llsun_rrsun_bolmag and return_and_show_entire_dataframe:
        
            if calculate_fiducial_model_radius: 

                if ell1_df_only:
        
                    return (results_df[results_df['l']==1], l_div_lsun, r_div_rsun,
                    Mbol_model, fiducial_model_radius)
        
                elif ell2_df_only:
        
                    return (results_df[results_df['l']==2], l_div_lsun, r_div_rsun,
                    Mbol_model, fiducial_model_radius)
                
                else:
                    
                    return (results_df, l_div_lsun, r_div_rsun, Mbol_model, fiducial_model_radius)

        # Case (T, T=default, F):

            else: 
    
                if ell1_df_only:

                    return (results_df[results_df['l']==1], l_div_lsun, r_div_rsun,
                    Mbol_model)
        
                elif ell2_df_only:

                    return (results_df[results_df['l']==2], l_div_lsun, r_div_rsun,
                    Mbol_model)
                    
                return (results_df, l_div_lsun, r_div_rsun, Mbol_model)

        # Case (T, F, T=default):
        
        elif return_llsun_rrsun_bolmag and not return_and_show_entire_dataframe:

            if calculate_fiducial_model_radius:
                
                return l_div_lsun, r_div_rsun, Mbol_model, fiducial_model_radius

        # Case (T, F, F): not a very useful case, but included for completeness. 
        
            else: 

                return l_div_lsun, r_div_rsun, Mbol_model

        # Case (F,T,T) and (F,T,F): 1 output (appropriate dataframe)
        
        elif return_and_show_entire_dataframe:
                
            if ell1_df_only:

                return results_df[results_df['l']==1]

            elif ell2_df_only:

                return results_df[results_df['l']==2]
            
            else: 

                # display(results_df)
            
                return results_df 

        # Case (F, F, T): 1 output (radius)
        
        elif not return_and_show_entire_dataframe:
            
            if tabulate_periods_and_modes:
                
                df = pd.DataFrame()
                header = [f"(l,k) = {lk}" for lk in modified_list_of_k]
           
                # for i, h in enumerate(header):
                    # df[h] = [periods[i]] # from the top

                df = pd.DataFrame({h: [p] for h,p in zip(header, periods)})
                
                df.index = df.index + 1 
                
                return df

            else:
            
                return fiducial_model_radius

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        print("Retrying...")

        with open(full_path, 'r') as file:

            lines = file.readlines()
            # print(lines)

            if len(lines) <=3 or not os.path.isfile(file): # reasonable to check for 3 lines

                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)

                print("Try running this function again. The model should now exist.")
    
            else:
        
                return np.NaN

def count_how_many_modes(file_ext):
    """ 
    This function is an extension of tabulate_results_dat. It is designed to count the 
    number of modes for both l=1 and l=2 that are presented in the results.dat file.
    It is helpful to have this function outside the previous function for modularity, 
    though there is overlap in some of the code.
    
    Input:

    Output:
    """

     # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The results.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired results.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.

        try: 
            
            model_params = parse_model_parameters(file_ext)
    
            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full results.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_results_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list
    
            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):
    
                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:
    
            # There might be some unforeseen error in parsing parameters, etc. 
    
            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):

    with open(full_path, 'r') as file:

        lines = file.readlines()

        # Instantiate the empty lists
        
        ell1 = []
        ell2 = []

        # Idea: each time a new l=1 or l=2 mode appears, append the ell value
        # to an empty list. At the end, count how many of each there are. 
        
        for i, line in enumerate(lines[9:]):

            ell = line.strip().split()[0] # all ells 
            ell = int(float(ell))

            if ell == 1:
                
                ell1.append(ell)

            if ell == 2:
                
                ell2.append(ell)

        num_ell1 = len(ell1)
        num_ell2 = len(ell2)
            
        return num_ell1, num_ell2
                    
def tabulate_chirt_dat(file_ext):
    """
    Input:
    - chirt.dat file extension (str)

    Output: Pandas dataframe (df)
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*) 
    - kapr: Thermodynamics gradient 
    - kapt: Thermodynamics gradient 
    """

    # Column names of the WDEC output file:
    
    headers = ['partt', 'kapr', 'kapt'] 

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The chirt.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired chirt.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full chirt.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_chirt_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
        
        chirt_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        chirt_df.index = chirt_df.index + 1

        return chirt_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def tabulate_check_dat(file_ext):
    """ 
    DOCSTRING COMING SOON
    """
    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The check.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired check.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full check.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_check_dat = True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None

    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):

    try:

        all_data = []
        first_set_temp_column_names = ['n', 'r/r*', 'y1', 'y2', 'tint', 'wint', 'placeholder']
        second_set_temp_column_names = ['n (again)', 'y3', 'y4', 't(i)', 'c(i)', 'n(i)', 'g(i)']
        
        with open(full_path, 'r') as file:

            lines = file.readlines()

            # The first data block starts at line 27
            
            for line in lines[26:]:
                
                if not 'guessed' in line and not 'ntry' in line and \
                not 'and' in line and not '*' in line and \
                not 'final' in line and not 'phase' in line and \
                not 'boundary' in line and not 'nodes' in line and \
                not 'y3' in line and not 'ekin' in line and \
                not 'kinetic' in line:
    
                    all_data.append(line.strip().split()) # splits into columns
            
            df = pd.DataFrame(all_data, columns = first_set_temp_column_names)
            df.set_index(df['n'], inplace=True) # simplest to index by shared values
            pd.set_option('display.max_rows', None)
            df.dropna(subset=['n'], inplace=True)

            # Create two (corresponding) smaller dataframes. Impose the simplest condition
            # possible to do so.

            small_df_1 = pd.DataFrame()
            small_df_2 = pd.DataFrame()

            # Easiest to sort by the presence of NaNs in the last colum
            
            if df['placeholder'].isna().any():

                small_df_1 = df[df['placeholder'].isna()]

            if df['placeholder'].notna().any():

                small_df_2 = df[df['placeholder'].notna()]

            # Instate different column names for the smaller dataframe
            
            small_df_2.columns = second_set_temp_column_names

            # Concatenate the two dataframe
            
            result = pd.concat([small_df_1, small_df_2], axis=1)
            
            return result
            
    except Exception as e:

        print(f"Error: {e}")
        
def tabulate_corsico_dat(file_ext):
    """
    Input:
    - corsico.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - xo: Oxygen mass fraction
    - xc: Carbon mass fraction
    - xhe: Helium mass fraction
    - xh: Hydrogen mass fraction
    - bled: “B Ledoux”
    - bvfreq: Brunt-Vaisala frequency (N^2)
    """

    # Column names of the WDEC output file:
    
    headers = ['partt', 'xo', 'xc', 'xhe', 'xh', 'bled', 'bvfreq'] 

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The corsico.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired corsico.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full corsico.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_corsico_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
        
        corsico_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers,
                                skiprows=1)

        corsico_df.index = corsico_df.index + 1 
            
        # These lines are only here to ensure consistency with the precision of the 
        # actual output file:
        
        # corsico_df.iloc[:, 0] = corsico_df.iloc[:, 0].astype(float).to_numpy()
        # corsico_df.iloc[:, 1] = corsico_df.iloc[:, 1].astype(float).to_numpy()
        # corsico_df.iloc[:, 2] = corsico_df.iloc[:, 2].astype(float).to_numpy()
        # corsico_df.iloc[:, 3] = corsico_df.iloc[:, 3].astype(float).to_numpy()
        # corsico_df.iloc[:, 4] = corsico_df.iloc[:, 4].astype(float).to_numpy()
        # corsico_df.iloc[:, 5] = corsico_df.iloc[:, 5].astype(float).to_numpy()
        # corsico_df.iloc[:, 6] = corsico_df.iloc[:, 6].astype(float).to_numpy()
        
        return corsico_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def tabulate_cpvtga1_dat(file_ext):
    """
    Input:
    - cpvtga1.dat file extension (str)

    Output: Pandas dataframe (df)
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - log(cv): Log heat capacity at constant volume (erg/K)
    - log(cp): Thermodynamics gradient
    - gam1: Thermodynamics gradient
    - fr: r/rstar
    """

    # Column names of the WDEC output file:
    
    headers = ['partt','log(cv)','log(cp)','gam1','fr'] 

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The cpvtga1.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired cpvtga1.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full cpvtga1.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_cpvtga1_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
        
        cpvtga1_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        cpvtga1_df.index = cpvtga1_df.index + 1
            
        return cpvtga1_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def tabulate_deld_dat(file_ext):
    """
    Input:
    - deld.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - del: Thermodynamics gradient (relates to convection)
    - delad: Thermodynamics gradient (relates to convection)
    """

    # Column names of the WDEC output file:
    
    headers = ['partt','del','delad'] 

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The deld.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired deld.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full corsico.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_deld_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
       
        deld_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        deld_df.index = deld_df.index + 1
            
        return deld_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def tabulate_epsrt_dat(file_ext):
    """
    Input:
    - epsrt.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - mr/mstar: Fractional mass coordinate
        - mr: Mass coordinate (mass inside the current shell, in grams)
        - mstar: Total mass of the star (in grams)
    - epsr: Thermodynamics gradient
    - epst: Thermodynamics gradient
    - log(eps): Log neutrino energy loss rate (ergs/s/g)
    - fr: r/rstar
    """

    # Column names of the WDEC output file:
    
    headers = ['mr/mstar','epsr','epst','log(eps)','fr'] 

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The epsrt.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired epsrt.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full epsrt.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_epsrt_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
        
        epsrt_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        epsrt_df.index = epsrt_df.index + 1
            
        # These lines are only here to ensure consistency with the precision of the 
        # actual output file:
        
        # epsrt_df.iloc[:,1] = epsrt_df.iloc[:,1].astype(float).to_numpy()
        # epsrt_df.iloc[:,2] = epsrt_df.iloc[:,2].astype(float).to_numpy()
        # epsrt_df.iloc[:,3] = epsrt_df.iloc[:,3].astype(float).to_numpy()
            
        return epsrt_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None
        
def tabulate_kaprt_dat(file_ext):
    """
    Input:
    - kaprt.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - chr: Thermodynamics gradient
    - cht: Thermodynamics gradient
    - log(kap): Log opacity (cm^2/g)
    - fr: r/rstar
    - xi: ln(r/p)
    - 1/delad: Inverse thermodynamics gradient (convection)
    - log(tth): Log thermal timescale at radius coordinate r (in cm)
    - gam3-1: Thermodynamics gradient
    """
    
    # Column names of the WDEC output file:
    
    headers = ['partt','chr','cht','log(kap)','fr','xi','1/delad','log(tth)','gam3-1']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The kaprt.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired kaprt.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full kaprt.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_kaprt_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)

        kaprt_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        kaprt_df.index = kaprt_df.index + 1
            
        return kaprt_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None
        
def tabulate_lrad_dat(file_ext):
    """
    Input:
    - lrad.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - log(lr): Log luminosity at radius coordinate r (ergs/s)
    - log(tth): Log thermal timescale at radius coordinate r (in cm)
    - xi: ln(r/p)
    """
    
    # Column names of the WDEC output file:
    
    headers = ['partt','log(lr)','log(tth)','xi']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The lrad.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired lrad.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full lrad.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_lrad_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
      
        lrad_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        lrad_df.index = lrad_df.index + 1
            
        return lrad_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def tabulate_prop_dat(file_ext):
    """
    Input:
    - prop.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - acous: Acoustic frequency
    - bvfreq: Brunt-Vaisala frequency
    - bvfrq: Brunt-Vaisala frequency (without bumps)
    - fr: r/rstar
    - xi: ln(r/p)
    - tfreq: Unsure 
    """
    
    # Column names of the WDEC output file:
    
    headers = ['partt','acous','bvfreq','bvfrq','fr','xi','tfreq']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The prop.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired prop.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full prop.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_prop_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)

        prop_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        prop_df.index = prop_df.index + 1
            
        return prop_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None
        
def tabulate_temprp_dat(file_ext):
    """
    Input:
    - temprp.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - log(T): Log temperature at radius r (Kelvin)
    - log(rho): Log density at radius r (g/cm^3)
    - log(p): Log pressure at radius r (dynes/cm^3)
    - fr: r/rstar
    - xi: ln(r/p)
    - log(lr): Log luminosity at radius r (erg/s)
    """
    
    # Column names of the WDEC output file:
    
    headers = ['partt','log(T)','log(rho)','log(p)','fr','xi','log(lr)']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The temprp.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired temprp.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full temprp.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_temprp_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
        
        temprp_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        temprp_df.index = temprp_df.index + 1 
            
        return temprp_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None
        
def tabulate_struc_dat(file_ext):
    """
    Input:
    - struc.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - n: Shell number (file output starts at n=2)
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - bvfreq: Brunt-Vaisala frequency
    - log(tth): Log thermal timescale at radius r
    - log(lr): Log luminosity at radius r (erg/s)
    - bvfrq: Brunt-Vaisala frequency (without bumps)
    - bvfreqmag: Scaled version of bvfreq (Brunt-Vaisala frequency)
    - mr/mstar: Fractional mass coordinate
        - mr: Mass coordinate (mass inside the current shell, in grams)
        - mstar: Total mass of the star (in grams)
    - xi: ln(r/p)
    """
    
    # Column names of the WDEC output file:
    
    headers = ['n','partt','bvfreq','log(tth)','log(lr)','bvfrq','bvfreqmag',
               'mr/mstar','xi']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The struc.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired struc.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full struc.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_struc_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
       
        struc_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers, )

        struc_df.index = struc_df.index + 1 
            
        # These lines are only here to ensure consistency with the precision of the 
        # actual output file:
        
        # struc_df.iloc[:,1] = struc_df.iloc[:,1].astype(float).to_numpy()
        # struc_df.iloc[:,2] = struc_df.iloc[:,2].astype(float).to_numpy()
        # struc_df.iloc[:,3] = struc_df.iloc[:,3].astype(float).to_numpy()
        # struc_df.iloc[:,4] = struc_df.iloc[:,4].astype(float).to_numpy()
        # struc_df.iloc[:,5] = struc_df.iloc[:,5].astype(float).to_numpy()
        # struc_df.iloc[:,6] = struc_df.iloc[:,6].astype(float).to_numpy()
        # struc_df.iloc[:,7] = struc_df.iloc[:,7].astype(float).to_numpy()
        # struc_df.iloc[:,8] = struc_df.iloc[:,8].astype(float).to_numpy()
            
        return struc_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None
        
def tabulate_xhe_dat(file_ext):
    """
    Input:
    - xhe.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - partt: Mass coordinate of the core = -log(1 – Mr/M*)
    - mr: Mass coordinate (mass inside the current shell, in grams)
    - xhe: Helium mass fraction
    - bled: “B Ledoux”
    - xo: Oxygen mass fraction
    - xc: Carbon mass fraction
    - fr: r/rstar
    - xh: Hydrogen mass fraction
    - log(r/p), see note below:
        - NB: There is a discrepancy between log vs ln. in the Fortran code and the user 
        manual. The user manual explicitly says "log(r/p)," not "xi." The Fortran code 
        says "xi," which suggests a natural logarithm (ln). 
    """
    # Column names of the WDEC output file:
    
    headers = ['partt','mr','xhe','bled','xo','xc','fr','xh','log(r/p)']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The xhe.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired xhe.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full xhe.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_xhe_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list

            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):

                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)

        xhe_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers)

        xhe_df.index = xhe_df.index + 1

        # These lines are only here to ensure consistency with the precision of the 
        # actual output file:

        # xhe_df.iloc[:,0] = xhe_df.iloc[:,0].astype(float).to_numpy()
        # xhe_df.iloc[:,1] = xhe_df.iloc[:,1].astype(float).to_numpy()
        # xhe_df.iloc[:,2] = xhe_df.iloc[:,2].astype(float).to_numpy()
        # xhe_df.iloc[:,3] = xhe_df.iloc[:,3].astype(float).to_numpy()
        # xhe_df.iloc[:,4] = xhe_df.iloc[:,4].astype(float).to_numpy()
        # xhe_df.iloc[:,5] = xhe_df.iloc[:,5].astype(float).to_numpy()
        # xhe_df.iloc[:,6] = xhe_df.iloc[:,6].astype(float).to_numpy()
        # xhe_df.iloc[:,7] = xhe_df.iloc[:,7].astype(float).to_numpy()
        # xhe_df.iloc[:,8] = xhe_df.iloc[:,8].astype(float).to_numpy()
            
        return xhe_df

    except Exception as e:

        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def tabulate_tape28_dat(file_ext):
    """
    Input:
    - tape28.dat file extension (str)

    Output: Pandas dataframe (df). 
    (Descriptions courtesy of Agnes Bischoff Kim, "Oth Order Introduction to Makedx.")
    - xi: ln(r/p)
    - r: Radius coordinate (cm) 
    - g: Acceleration due to gravity at radius r (cm/s^2)
    - rho: Density at radius r (g/cm^3)
    - mr: Mass coordinate (mass inside the current shell, in grams)
    """
    # Column names of the WDEC output file:
    
    headers = ['xi','r','g','rho','mr']

    # Relevant paths:
    
    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
    full_path = path + file_ext

    if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:

    # The tape28.dat file may or may not exist in future function calls. This 
    # can happen if a string (with the desired tape28.dat file path) is fed into
    # the function, but for which no actual file contents exist. In this case, the 
    # model should be computed first, and the output files should be saved. It could
    # also be (for some weird reason) that the appropriate file name exists and was
    # saved at some point with a .txt file format, but the actual file is empty.
        
        try: 
            
            model_params = parse_model_parameters(file_ext)

            if not os.path.isfile(full_path):
                
                compute_wdec_model(*model_params)
                name_and_save_wdec_output_files(*model_params)
            
            # All the files are written and saved at this step. In particular, the 
            # full tape28.dat path is saved here (in memory): 
                
            full_path = name_and_save_wdec_output_files(*model_params, need_tape28_dat=True)
            full_path = full_path[0] # technically the first/only element (str) of the list
            
            # Error message raised just in case there are issues in creating the model (for
            # whatever weird reason):
            
            if not os.path.isfile(full_path):
                
                raise FileNotFoundError(f"An attempt was made to create the model: {model_params}."
                                       f"The model was not successfully created, and the file "\
                                       f"{full_path} does not exist.")
            
        except Exception as e:

            # There might be some unforeseen error in parsing parameters, etc. 

            print(f"Error: {e}")
            
            return None
            
    # Now proceed to the code that will handle the majority of cases (i.e., for existing
    # files):
       
    try: 

        if os.path.isfile(full_path):
        
            model_params = parse_model_parameters(file_ext)
      
        # Skip some rows of text (extra info, irrelevant)
        tape28_df = pd.read_csv(full_path,delimiter=r'\s+', names=headers, skiprows=3)

        tape28_df.index = tape28_df.index + 1
        
        # These lines are only here to ensure consistency with the precision of the 
        # actual output file:
        
        # tape28_df.iloc[:,0] = tape28_df.iloc[:,0].astype(float).to_numpy()
        # tape28_df.iloc[:,1] = tape28_df.iloc[:,1].astype(float).to_numpy()
        # tape28_df.iloc[:,2] = tape28_df.iloc[:,2].astype(float).to_numpy()
        # tape28_df.iloc[:,3] = tape28_df.iloc[:,3].astype(float).to_numpy()
        # tape28_df.iloc[:,4] = tape28_df.iloc[:,4].astype(float).to_numpy()
            
        return tape28_df

    except Exception as e:
        
        print(f"The file {full_path} was read in properly, but the following error has " \
             f"occurred: {e}")

        return None

def make_a_plot(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100, abundance_vs_mass_coord=False,
        oxgyen_abundance_vs_mmsun = False, bv_vs_mass_coord = False,
        lamb_vs_mass_coord = False, log_KE_vs_period_ell1 = False, 
        log_KE_vs_period_ell2 = False, ckl_vs_period_ell1 = False, 
        ckl_vs_period_ell2 = False, log_kappa_vs_rrstar = False, 
        ):
    """
    This function will call on any of the individual functions above to generate 
    some of the more common/useful plots of interest. This list is growing, so 
    additional changes will be added as appropriate. This function is designed to 
    generate a single plot; if multiple plots are desired, the function should be 
    called sequentially (with different boolean values, as appropriate). 

    Input:
    - all model parameters (ints and/or floats)
    - type of plot to generate (bool)
    """

    if abundance_vs_mass_coord:

        # Creates the correct file name for the given parameters
        end_ext = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100, name='corsico')

        corsico_df = tabulate_corsico_dat(end_ext) # see doc string for description
            
        # Read in relevant columns
        x = corsico_df['partt'].astype(float).to_numpy()
        y1 = corsico_df['xo'].astype(float).to_numpy()
        y2 = corsico_df['xc'].astype(float).to_numpy()
        y3 = corsico_df['xhe'].astype(float).to_numpy()
        y4 = corsico_df['xh'].astype(float).to_numpy()

        # Plot data
        plt.figure()
        
        plt.plot(x, y1, label='Oxygen Abundance')
        plt.plot(x, y2, label='Carbon Abundance')
        plt.plot(x, y3, label='Helium Abundance')
        plt.plot(x, y4, label='Hydrogen Abundance')

        plt.xlabel(xlabel=format_names_properly('partt'))
        plt.ylabel(ylabel='Chemical Abundance')
        plt.legend(loc='best')

        plt.show()
        plt.close()
    
    if oxgyen_abundance_vs_mmsun:
        
        # Creates the correct file name for the given parameters
        end_ext = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100, name='corsico')

        corsico_df = tabulate_corsico_dat(end_ext) # see doc string for description

        # Convert to M/M* axis scale
        x_old = corsico_df['partt'].astype(float).to_numpy()
        x_mmsun = 1-10**-x_old
        y = corsico_df['xo'].astype(float).to_numpy()

        # Plot data
        plt.figure()
        
        plt.plot(x_mmsun, y)
        plt.xlabel(xlabel=format_names_properly('mmsun'))
        plt.ylabel(ylabel='Oxygen Abundance')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        plt.show()
        plt.close()

    # Propagation diagram

    if bv_vs_mass_coord or lamb_vs_mass_coord:

        # Creates the correct file name for the given parameters
        end_ext = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100, name='prop')

        prop_df = tabulate_prop_dat(end_ext) # see doc string for description

        # Mass coordinate on x-axis (in either case)
        x = prop_df['partt'].astype(float).to_numpy()
        
        if bv_vs_mass_coord and lamb_vs_mass_coord:
        
            bv_old = prop_df['bvfreq'].astype(float).to_numpy() # Log(N^2)
            lamb_old = prop_df['acous'].astype(float).to_numpy() # Log(L^2)

            # Appropriately scaled BV and Lamb frequencies (accounts for both positive 
            # and negative frequencies)
            bv_new = 10**(bv_old) 
            lamb_new = 10**(lamb_old)

            # Plot data
            plt.figure()
            
            plt.plot(x, bv_new, label=format_names_properly('bvfreq'))
            plt.plot(x, lamb_new, label=format_names_properly('acous'))
            
            plt.xlabel(xlabel=format_names_properly('partt'))
            plt.ylabel(ylabel=format_names_properly('ang_freq'))

            # Preliminary bounds
            bv_min = np.min(np.abs(bv_new))
            bv_max = np.max(np.abs(bv_new))
            lamb_min = np.min(np.abs(lamb_new))
            lamb_max = np.max(np.abs(lamb_new))

            y_min_adjustment = 10*abs(min(bv_min, lamb_min))
            y_max_adjustment = 20*abs(max(bv_max, lamb_max))
            
            y_min = min(bv_min, lamb_min) - y_min_adjustment
            y_max = max(bv_max, lamb_max) + y_max_adjustment

            # Axis bounds
            plt.yscale('log')
            plt.ylim(y_min, y_max)
            plt.legend(loc='best')

            plt.show()
            plt.close()
    
        elif bv_vs_mass_coord:
    
            # Read in relevant columns
            y_old = prop_df['bvfreq'].astype(float).to_numpy() # Log(N^2)
            y_new = 10**(y_old)
            
            # Plot data
            plt.figure()
            
            plt.xlabel(xlabel=format_names_properly('partt'))
            plt.ylabel(ylabel=format_names_properly('ang_freq'))
            plt.yscale('log')

            y_min_adjustment = 0.10*abs(min(y_new))
            y_max_adjustment = 40*abs(max(y_new))

            plt.ylim(np.min(y_new)-y_min_adjustment, np.max(y_new)+y_max_adjustment) 
            plt.plot(x, y_new)

            plt.show()
            # plt.close()

        elif lamb_vs_mass_coord:
    
            # Read in relevant columns
            y_old = prop_df['acous'].astype(float).to_numpy() # Log(L^2)
            y_new = 10**(y_old)

            # Plot data
            plt.figure()
            
            plt.xlabel(xlabel=format_names_properly('partt'))
            plt.ylabel(ylabel=format_names_properly('ang_freq'))
            plt.yscale('log')
            
            y_min_adjustment = 0.25*abs(min(y_new))
            y_max_adjustment = 10*abs(max(y_new))

            plt.ylim(np.min(y_new)-y_min_adjustment, np.max(y_new)+y_max_adjustment) 
            plt.plot(x, y_new)

            plt.show()
            # plt.close()

    if log_KE_vs_period_ell1 or log_KE_vs_period_ell2: 

        # Creates the correct file name for the given parameters
        end_ext = unique_results_dat_file_id(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100)

        if log_KE_vs_period_ell1 and log_KE_vs_period_ell2:
            
            # Results.dat dataframes (both l=1 and l=2)
            results_df_ell1 = tabulate_results_dat(end_ext, ell1_df_only=True)
            results_df_ell2 = tabulate_results_dat(end_ext, ell2_df_only=True)

            # Read in relevant columns
            x_ell1 = results_df_ell1['Period (s)'].astype(float).to_numpy()
            y_ell1 = results_df_ell1['Log KE'].astype(float).to_numpy()
            
            x_ell2 = results_df_ell2['Period (s)'].astype(float).to_numpy()
            y_ell2 = results_df_ell2['Log KE'].astype(float).to_numpy()

            # Plot data
            plt.figure()
            
            plt.plot(x_ell1, y_ell1, label=format_names_properly('ell1'))
            plt.plot(x_ell2, y_ell2, label=format_names_properly('ell2'))

            plt.xlabel(xlabel=f"Period {format_names_properly('Pk')}")
            plt.ylabel(ylabel=format_names_properly('LogEkink'))
            plt.legend(loc='best') 

            plt.show()
            plt.close()

        elif log_KE_vs_period_ell1: 
    
            # Results.dat dataframe (l=1)
            results_df_ell1 = tabulate_results_dat(end_ext, ell1_df_only=True)
    
            # Read in relevant columns
            x_ell1 = results_df_ell1['Period (s)'].astype(float).to_numpy()
            y_ell1 = results_df_ell1['Log KE'].astype(float).to_numpy()

            # Plot data
            plt.figure()
            
            plt.plot(x_ell1, y_ell1, label=format_names_properly('ell1'))
            plt.xlabel(xlabel=f"Period {format_names_properly('Pk')}")
            plt.ylabel(ylabel=format_names_properly('LogEkink'))
            plt.legend(loc='best') 

            plt.show()
            plt.close()
            
        elif log_KE_vs_period_ell2: 
    
            # Results.dat dataframe (l=2)
            results_df_ell2 = tabulate_results_dat(end_ext, ell2_df_only=True)
    
            # Read in relevant columns
            x_ell2 = results_df_ell2['Period (s)'].astype(float).to_numpy()
            y_ell2 = results_df_ell2['Log KE'].astype(float).to_numpy()

            # Plot data
            plt.figure()
            
            plt.plot(x_ell2, y_ell2, label=format_names_properly('ell2'))
            plt.xlabel(xlabel=f"Period {format_names_properly('Pk')}")
            plt.ylabel(ylabel=format_names_properly('LogEkink'))
            plt.legend(loc='best') 

            plt.show()
            plt.close()
            
    if ckl_vs_period_ell1 or ckl_vs_period_ell2:

        # Creates the correct file name for the given parameters
        end_ext = unique_results_dat_file_id(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100)

        if ckl_vs_period_ell1 and ckl_vs_period_ell2:
            
            # Results.dat dataframes (both l=1 and l=2)
            results_df_ell1 = tabulate_results_dat(end_ext, ell1_df_only=True)
            results_df_ell2 = tabulate_results_dat(end_ext, ell2_df_only=True)
    
            # Read in relevant columns
            x_ell1 = results_df_ell1['Period (s)'].astype(float).to_numpy()
            y_ell1 = results_df_ell1['ckl'].astype(float).to_numpy()

            x_ell2 = results_df_ell2['Period (s)'].astype(float).to_numpy()
            y_ell2 = results_df_ell2['ckl'].astype(float).to_numpy()

            # Plot data
            plt.figure()
            
            plt.plot(x_ell1, y_ell1, label=format_names_properly('ell1'))
            plt.axhline(y=0.5, linestyle='--', linewidth=0.75) # asymptote = 1/2
            
            plt.plot(x_ell2, y_ell2, label=format_names_properly('ell2'))
            plt.axhline(y=0.16667, color='orange', linestyle='--', linewidth=0.75) 
            # asymptote = 1/6
            
            plt.xlabel(xlabel=f"Period {format_names_properly('Pk')}")
            plt.ylabel(ylabel=format_names_properly('ckl'))
            plt.legend(loc='best')

            plt.show()
            plt.close()

        elif ckl_vs_period_ell1:
    
            # Creates the correct file name for the given parameters
            end_ext = unique_results_dat_file_id(Teff, Mass, Menv, Mhe, Mh, 
            He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
            diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
            w1x100, w2x100, w3x100, w4x100)
    
            # Results.dat dataframe (l=1)
            results_df_ell1 = tabulate_results_dat(end_ext, ell1_df_only=True)
    
            # Read in relevant columns
            x_ell1 = results_df_ell1['Period (s)'].astype(float).to_numpy()
            y_ell1 = results_df_ell1['ckl'].astype(float).to_numpy()

            # Plot data
            plt.figure()
            
            plt.plot(x_ell1, y_ell1, label=format_names_properly('ell1'))
            plt.axhline(y=0.5, linestyle='--', linewidth=0.75) # asymptote = 1/2
            plt.xlabel(xlabel=f"Period {format_names_properly('Pk')}")
            plt.ylabel(ylabel=format_names_properly('ckl'))
            plt.legend(loc='best') 

            plt.show()
            plt.close()
            
        elif ckl_vs_period_ell2:
    
            # Creates the correct file name for the given parameters
            end_ext = unique_results_dat_file_id(Teff, Mass, Menv, Mhe, Mh, 
            He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
            diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
            w1x100, w2x100, w3x100, w4x100)
    
            # Results.dat dataframe (l=2)
            results_df_ell2 = tabulate_results_dat(end_ext, ell2_df_only=True)
    
            # Read in relevant columns
            x_ell2 = results_df_ell2['Period (s)'].astype(float).to_numpy()
            y_ell2 = results_df_ell2['ckl'].astype(float).to_numpy()

            # Plot data
            plt.figure()
            
            plt.plot(x_ell2, y_ell2, label=format_names_properly('ell2'))
            plt.axhline(y=0.16667,linestyle='--', linewidth=0.75) # asymptote = 1/6
            plt.xlabel(xlabel=f"Period {format_names_properly('Pk')}")
            plt.ylabel(ylabel=format_names_properly('ckl'))
            plt.legend(loc='best')  

            plt.show()
            plt.close()

    if log_kappa_vs_rrstar:

        # Creates the correct file name for the given parameters
        end_ext = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, 
        He_abund_mixed_CHeH_region, diff_coeff_He_base_env,
        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, h3x100,
        w1x100, w2x100, w3x100, w4x100, name='kaprt')

        kaprt_df = tabulate_kaprt_dat(end_ext) # see doc string for description
        
        # Read in relevant columns
        x = kaprt_df['fr'].astype(float).to_numpy() 
        # SHOULDN'T THE DECIMAL BE ONE OVER TO THE LEFT FOR R/R*? INSPECT DATA TABLE. 
        y = kaprt_df['log(kap)'].astype(float).to_numpy()

        # Plot data
        plt.figure()
        
        plt.xlabel(xlabel=format_names_properly('fr'))
        plt.ylabel(ylabel=format_names_properly('log_kappa'))
        plt.plot(x,y)

        plt.show()
        plt.close()

class specific_model:
    """
    This class is designed for easy access to functions that output the individual
    WDEC parameters:
    - Teff
    - Mass
    ... (etc.) 
    
    In addition, it contains functions that can access quick information, including:
    - results.dat file extension (str)
    - r/rsun (float)
    - l/lsun (float)
    - bolometric WDEC model absolute magnitude, Mbol (float)
    - model radius, in cm (float)
    """
    
    def __init__(self, results_dat_tag_filename):
    
        path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
        full_path_results_dot_dat = path + results_dat_tag_filename
        
        self.results_dat_tag_filename = results_dat_tag_filename
        self.results_dot_dat_full_filename = full_path_results_dot_dat

        # Compute the model if it does not yet exist. Save the output files.

        if not os.path.isfile(full_path_results_dot_dat):

            # For a provided, non-existent results.dat full file path (i.e., 
            # non-existent WDEC model), compute the model and save the results. 
            
            desired_model = parse_model_parameters(results_dat_tag_filename)
    
            compute_wdec_model(*desired_model)

            # All output files are saved and written in this step. In particular, the
            # results.dat file name is stored in memory:
            
            full_path_results_dot_dat = name_and_save_wdec_output_files(*desired_model, 
                                                                        need_results_dat=True)
            
        # Extract the parameters from the existing (or newly created) file 
        # name.
        
        (
        self.Teff, self.Mass, self.Menv, self.Mhe, self.Mh, 
        self.He_abund_mixed_CHeH_region, self.diff_coeff_He_base_env, 
        self.diff_coeff_He_base_pure_He, self.alpha, self.h1x100, 
        self.h2x100, self.h3x100, self.w1x100, self.w2x100, self.w3x100, 
        self.w4x100 
        ) = parse_model_parameters(self.results_dat_tag_filename)
        
    ########################################################################
    # The following is a series of functions to simply output the model
    # parameter of interest.

    def get_teff(self):

        return self.Teff
    
    def get_mass(self):

        return self.Mass
    
    def get_menv(self):

        return self.Menv
    
    def get_mhe(self):

        return self.Mhe
    
    def get_mh(self):

        return self.Mh
    
    def get_He_abund_mixed_CHeH_region(self): 

        return self.He_abund_mixed_CHeH_region
    
    def get_diff_coeff_He_base_env(self):

        return self.diff_coeff_He_base_env
    
    def get_diff_coeff_He_base_pure_He(self):

        return self.diff_coeff_He_base_pure_He
    
    def get_alpha(self):

        return self.alpha
    
    def get_h1x100(self):

        return self.h1x100
    
    def get_h2x100(self):

        return self.h2x100
    
    def get_h3x100(self):

        return self.h3x100
    
    def get_w1x100(self):

        return self.w1x100
    
    def get_w2x100(self):

        return self.w2x100
    
    def get_w3x100(self):

        return self.w3x100
    
    def get_w4x100(self):

        return self.w4x100

    def get_results_dat_file_ext(self):

        # Seems unnecessary, but this is actually useful for future function
        # calls.
        
        return self.results_dat_tag_filename
    
    def get_llsun(self):
        
        self.llsun, _, _, _ = tabulate_results_dat(
            file_ext = self.results_dat_tag_filename, 
            return_llsun_rrsun_bolmag = True,
            return_and_show_entire_dataframe = False, 
            calculate_fiducial_model_radius = True)
        
        return self.llsun
    
    def get_rrsun(self):
        
        _, self.rrsun, _, _ = tabulate_results_dat(
            file_ext = self.results_dat_tag_filename, 
            return_llsun_rrsun_bolmag = True,
            return_and_show_entire_dataframe = False, 
            calculate_fiducial_model_radius = True)
        
        return self.rrsun
    
    def get_bolometric_magnitude(self):
        
        _, _, self.bolmag, _ = tabulate_results_dat(
            file_ext = self.results_dat_tag_filename, 
            return_llsun_rrsun_bolmag = True,
            return_and_show_entire_dataframe = False, 
            calculate_fiducial_model_radius = True)
        
        return self.bolmag
    
    def get_radius(self):
        
        self.radius = tabulate_results_dat(
            file_ext = self.results_dat_tag_filename,
            return_llsun_rrsun_bolmag = False,
            return_and_show_entire_dataframe = False,
            calculate_fiducial_model_radius = True)
    
        return self.radius  

def compute_a_model_and_save_results(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
            diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, 
            h3x100, w1x100, w2x100, w3x100, w4x100, num_dof = None, return_sigma_rms = False,
            without_abs_mag = False, with_abs_mag = False, 
            measured_period_array = None, measured_period_uncertainty_array = None, 
            include_absolute_magnitude = False, Bergeron_filter_if_applicable = None, 
            measured_absolute_magnitude = None, measured_absolute_magnitude_uncertainty = None, 
            include_detailed_error_messages = False
            ):
    """
    This function creates a WDEC model and saves the results, based on functions that 
    were previously written. It is a quick, convenient function going forward, though 
    the general structure of the function was implemented in previous code. The input
    parameters for this function are needed by other functions therein. 

    Input (required): 
    - all model parameters (ints and/or floats)

    Optional input:
    - number of degrees of freedom (int); default None
    - return_sigma_rms (boolean); default False

    Note: By default, this function is not meant to calculate a chi-squared value, especially 
    considering computational memory efficiency in higher dimensions. However, this function 
    provides such flexibility and can compute both chi-squared and reduced chi-squared values.
    Keep in mind that if a reduced chi-squared value is to be calculated, the user must specify
    the number of degrees of freedom. 

    Note: If a chi-squared calculation is to be performed, only a single calculation can be
    performed at a time (i.e., either with or without absolute magnitude). Setting both to 
    "True" or both to "False" will return None (all files associated with the fiducial model
    will, of course, have been created and saved nonetheless). 

    Note: This function also provides the flexibility to return the sigma rms value that can 
    (optionally) be calculated via the function calc_s. In this case, the output would be a 
    tuple of values (see docstring for the function calc_s). 

    Output:
    - NoneType (no chi-squared calculations)
    - int/float (chi-squared calculations)
    """

    # Note: To the best of my knowledge, the file 'radii.dat' does not contain anything.  
    
    all_file_tag_names = ['calcperiods', 'check', 'chirt', 'corsico',
                          'cpvtga1', 'deld', 'discr', 'epsrt',
                          'gridparameters', 'kaprt', 'lrad', 'modelp',
                          'output', 'pform', 'prop', 'radii', 'reflection2', 
                          'results', 'struc', 'tape18', 'tape19', 'tape28', 
                          'tape29', 'temprp', 'thorne', 'xhe', 'xir']

    # Check to see if every output file exists (already) for the desired model (input
    # parameters). If they do, the code will skip right over any new model computations
    # and use the data from the existing files. 

    path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'

    all_file_full_paths = []
    
    for tag_name in all_file_tag_names:

        # Generates the appropriate file name (e.g., "results.dat_..." or "corisco.dat...", etc.):
        
        file_id = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, h1x100, 
                            h2x100, h3x100, w1x100, w2x100, w3x100, w4x100, name = tag_name)
    
        full_path = path + file_id

        # Append all these (full) file names to a list. We will loop over these next. 
        
        all_file_full_paths.append(full_path)
    
    try:

        file_exists = []
        
        for output_file in all_file_full_paths:
    
            if 'radii' not in output_file: 
                
                # The file needs to exist in name and file size.
                    
                if os.path.isfile(output_file) and os.path.getsize(output_file) != 0:
                                            
                    # Append the truth value (to ensure both conditions are met before proceeding):
            
                    file_exists.append(True) 
    
                else:
    
                    file_exists.append(False)
    
            elif 'radii' in output_file:
    
                # To the best of my knowledge, the radii.dat file does not contain anything. 
                # However, it will be named and saved for completeness, just in case it does.
                # Notice the relaxation on the conditional statement used previously: 
    
                if os.path.isfile(output_file):
    
                    file_exists.append(True)
    
                else:
    
                    file_exists.append(False)
        
        # If there are no output files (or missing ones), we will need to create a (new) model.
    
        file_exists = np.array(file_exists)
        
        if not file_exists.any() or not file_exists.all(): 
            
            # First condition: true if all booleans are false
            # Second condition: true if at least one element is false
            
            # Compute the model
            compute_wdec_model(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
                            diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, 
                            h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, w4x100)
            
            # Save all output file names
            name_and_save_wdec_output_files(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
                            diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, 
                            h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, w4x100,
                            need_calcperiods=True, need_check_dat=True, need_chirt_dat=True,
                            need_corsico_dat=True, need_cpvtga1_dat=True, need_deld_dat=True,
                            need_discr_dat=True, need_epsrt_dat=True, need_gridparameters=True,
                            need_kaprt_dat=True, need_lrad_dat=True, need_modelp_dat=True,
                            need_output_dat=True, need_pform_dat=True, need_prop_dat=True, 
                            need_radii_dat=True, need_reflection2_dat=True, need_results_dat=True,
                            need_struc_dat=True, need_tape18_dat=True, need_tape19_dat=True,
                            need_tape28_dat=True, need_tape29_dat=True, need_temprp_dat=True)
        else:

            # print("The relevant file(s) already exist(s). No new model was computed.")

            pass 
            
    except Exception as e:

        print(f"Error: {e}")

    finally:

        # The model periods can be read in using the "read_in_calcperiods" function:
        
        calcperiods_file_tag = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, 
                        He_abund_mixed_CHeH_region, diff_coeff_He_base_env, 
                        diff_coeff_He_base_pure_He, alpha, h1x100, h2x100, 
                        h3x100, w1x100, w2x100, w3x100, w4x100, name='calcperiods')

        all_ell_model_periods_array = read_in_calcperiods(calcperiods_file_tag)
        
        # Case 1: Consider absolute magnitude.
        
        if with_abs_mag and not without_abs_mag:
            
            # Reduced chi-squared value or chi-squared value returned (only).

            if not return_sigma_rms:
                
                chi_sq = stats.calc_s(measured_period_array = measured_period_array, 
                            model_period_array = all_ell_model_periods_array, 
                            measured_period_uncertainty_array = measured_period_uncertainty_array, 
                            num_dof = num_dof, include_absolute_magnitude = True, 
                            Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                            measured_absolute_magnitude = measured_absolute_magnitude,
                            measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                            include_detailed_error_messages = include_detailed_error_messages, 
                            return_sigma_rms = False, Teff = Teff, Mass = Mass, Menv = Menv, 
                            Mhe = Mhe, Mh = Mh, He_abund_mixed_CHeH_region = He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env = diff_coeff_He_base_env, 
                            diff_coeff_He_base_pure_He = diff_coeff_He_base_pure_He, 
                            alpha = alpha, h1x100 = h1x100, h2x100 = h2x100, h3x100 = h3x100, w1x100 = w1x100, 
                            w2x100 = w2x100, w3x100 = w3x100, w4x100 = w4x100)
    
                return chi_sq

            # Both the (reduced) chi-squared & sigma rms values are returned
            
            elif return_sigma_rms:

                chi_sq, sigma_rms = stats.calc_s(measured_period_array = measured_period_array, 
                            model_period_array = all_ell_model_periods_array, 
                            measured_period_uncertainty_array = measured_period_uncertainty_array, 
                            num_dof = num_dof, include_absolute_magnitude = True, 
                            Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                            measured_absolute_magnitude = measured_absolute_magnitude,
                            measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                            include_detailed_error_messages = include_detailed_error_messages, 
                            return_sigma_rms = True, Teff = Teff, Mass = Mass, Menv = Menv, 
                            Mhe = Mhe, Mh = Mh, He_abund_mixed_CHeH_region = He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env = diff_coeff_He_base_env, 
                            diff_coeff_He_base_pure_He = diff_coeff_He_base_pure_He, 
                            alpha = alpha, h1x100 = h1x100, h2x100 = h2x100, h3x100 = h3x100, w1x100 = w1x100, 
                            w2x100 = w2x100, w3x100 = w3x100, w4x100 = w4x100)

                return chi_sq, sigma_rms
                
        # Case 2: Do not consider absolute magnitude.
        
        elif without_abs_mag and not with_abs_mag:

            # Reduced chi-squared value or chi-squared value returned (only).

            if not return_sigma_rms:
                
                chi_sq = stats.calc_s(measured_period_array = measured_period_array, 
                            model_period_array = all_ell_model_periods_array, 
                            measured_period_uncertainty_array = measured_period_uncertainty_array, 
                            num_dof = num_dof, include_absolute_magnitude = False, 
                            Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                            measured_absolute_magnitude = measured_absolute_magnitude,
                            measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                            include_detailed_error_messages = include_detailed_error_messages, 
                            return_sigma_rms = False, Teff = Teff, Mass = Mass, Menv = Menv, 
                            Mhe = Mhe, Mh = Mh, He_abund_mixed_CHeH_region = He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env = diff_coeff_He_base_env, 
                            diff_coeff_He_base_pure_He = diff_coeff_He_base_pure_He, 
                            alpha = alpha, h1x100 = h1x100, h2x100 = h2x100, h3x100 = h3x100, w1x100 = w1x100, 
                            w2x100 = w2x100, w3x100 = w3x100, w4x100 = w4x100)
    
                return chi_sq

            # Both the (reduced) chi-squared & sigma rms values are returned 
            
            elif return_sigma_rms:
                
                chi_sq, sigma_rms = stats.calc_s(measured_period_array = measured_period_array, 
                            model_period_array = all_ell_model_periods_array, 
                            measured_period_uncertainty_array = measured_period_uncertainty_array, 
                            num_dof = num_dof, include_absolute_magnitude = False, 
                            Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                            measured_absolute_magnitude = measured_absolute_magnitude,
                            measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                            include_detailed_error_messages = include_detailed_error_messages, 
                            return_sigma_rms = True, Teff = Teff, Mass = Mass, Menv = Menv, 
                            Mhe = Mhe, Mh = Mh, He_abund_mixed_CHeH_region = He_abund_mixed_CHeH_region,
                            diff_coeff_He_base_env = diff_coeff_He_base_env, 
                            diff_coeff_He_base_pure_He = diff_coeff_He_base_pure_He, 
                            alpha = alpha, h1x100 = h1x100, h2x100 = h2x100, h3x100 = h3x100, w1x100 = w1x100, 
                            w2x100 = w2x100, w3x100 = w3x100, w4x100 = w4x100)

                return chi_sq, sigma_rms
                
        else:

            # No calculations are performed by default, i.e., for "with" and "without" boolean 
            # combinations of (T,T) or (F,F). Only the "default operations" are performed 
            # (i.e., computing the model and saving the output files). 
            
            pass 

def plot_chi_squared_vs_varied_param(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region, 
                    diff_coeff_He_base_env, diff_coeff_He_base_pure_He, alpha, 
                    h1x100, h2x100, h3x100, w1x100, w2x100, w3x100, w4x100,
                    measured_period_array, measured_period_uncertainty_array, num_dof=None,
                    Bergeron_filter_if_applicable=None, measured_absolute_magnitude=None,
                    measured_absolute_magnitude_uncertainty=None, 
                    include_detailed_error_messages=False, plot_sigma_rms=False,
                    without_mag = True, with_mag = False, name_of_varied_param = None,
                    interval_of_varied_param = None, save_chi_sq_plot = False,
                    save_sigma_rms_plot = False):
    """
    This function is intended to be used to make a chi-squared or reduced chi-squared plot. 
    Specifically, it can be used to generate a chi-squared plot for a single varied parameter 
    on a specified interval, with all other parameters held fixed. 
    
    Note: This function provides the flexibility to produce a plot of the sigma rms values. When
    the input parameter "plot_sigma_rms" is set to True, both the chi-squared and sigma_rms plots
    will be returned sequentially. 

    IMPORTANT: This function will only work if the user specifies both the (single) varied
    parameter name along with the array of values over which the parameter should be varied. 
    The starting value of the varied parameter should be passed as the input value. (For 
    example, if you vary total mass starting from 0.6M, then you should pass 'Mass=600' to the 
    function call.)

    Input:
    - TBD
    
    Optional Input:
    - (total) number of degrees of freedom. If used as part of a nested for-loop in which 
    multiple parameters are varied, then this quantity should reflect the total number of
    varied parameters. Otherwise, if used to vary a single parameter, this should be set to one.

    Output:
    - TBD
    """

    # File destinations (used below for saving plots). Indicate the number of degrees of
    # freedom in the file name to clearly indicate whether the file was created as part 
    # of a nested sequence of trials. 
    
    plot_subdir = '/Users/adublin/Desktop/WDEC Trials and Summaries/Plots'

    save_path_chi_sq = os.path.join(plot_subdir, f"Chi-Squared vs. {name_of_varied_param} with "
                             f"num_dof={num_dof}.png")
    full_path_chi_sq = os.path.join(plot_subdir, save_path_chi_sq)

    save_path_sigma_rms = os.path.join(plot_subdir, f"Sigma rms vs. {name_of_varied_param} with "
                             f"num_dof={num_dof}.png")

    full_path_sigma_rms = os.path.join(plot_subdir, save_path_sigma_rms)
    
    # Instatiate the initial dictionary. The relevant parameter can be updated
    # each time (within the for loop). 
    
    kwargs_specific_iter = {'Teff': Teff, 'Mass': Mass, 'Menv': Menv, 'Mhe': Mhe,
            'Mh': Mh, 'He_abund_mixed_CHeH_region': He_abund_mixed_CHeH_region,
            'diff_coeff_He_base_env': diff_coeff_He_base_env,
            'diff_coeff_He_base_pure_He': diff_coeff_He_base_pure_He,
            'alpha': alpha, 'h1x100': h1x100, 'h2x100': h2x100, 'h3x100': h3x100,
            'w1x100': w1x100, 'w2x100': w2x100,'w3x100': w3x100, 'w4x100': w4x100}

    """ NEW: NEED TO MODIFY 

        # We will need to create the model from scratch in this case in order to produce
        # the list of model periods.

        fname = unique_specific_file_id(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                           diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                           alpha, h1x100, h2x100, h3x100, w1x100, w2x100,
                           w3x100, w4x100, name = 'calcperiods')

        path = '/Users/adublin/Desktop/WDEC/wdec-master/runs/'
        full_path = path + str(fname)
            
        # Compute the model 
        compute_wdec_model(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                           diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                           alpha, h1x100, h2x100, h3x100, w1x100, w2x100,
                           w3x100, w4x100)

        # Save the calcperiods results 
        name_and_save_wdec_output_files(Teff, Mass, Menv, Mhe, Mh, He_abund_mixed_CHeH_region,
                        diff_coeff_He_base_env, diff_coeff_He_base_pure_He,
                        alpha, h1x100, h2x100, h3x100, w1x100, w2x100,
                        w3x100, w4x100, need_calcperiods=True, need_check_dat=False,
                        need_chirt_dat=False, need_corsico_dat=False, need_cpvtga1_dat=False,
                        need_deld_dat=False, need_discr_dat=False, need_epsrt_dat=False,
                        need_gridparameters=False, need_kaprt_dat=False, need_lrad_dat=False,
                        need_modelp_dat=False, need_output_dat=False, need_pform_dat=False,
                        need_prop_dat=False, need_radii_dat=False, need_reflection2_dat=False,
                        need_results_dat=False, need_struc_dat=False, need_tape18_dat=False,
                        need_tape19_dat=False, need_tape28_dat=False, need_tape29_dat=False,
                        need_temprp_dat=False, need_thorne_dat=False, need_xhe_dat=False,
                        need_xir_dat=False) 

        # Store the periods
        model_period_array = read_in_calcperiods(calcperiods_file_ext = fname)

    """

    if name_of_varied_param is not None and interval_of_varied_param is not None:

        interval_of_varied_param = np.array(interval_of_varied_param)

        # Instantiate the relevant lists:
        
        chi_squared_no_mag = []
        chi_squared_with_mag = []

        # Since sigma_rms is not always desired (returned), it is best for memory
        # purposes only to create this list if specified to do so by the function
        # call. 
        
        if plot_sigma_rms:
            
            sigma_rms_no_mag = []
            sigma_rms_with_mag = []
        
        for specific_varied_value in interval_of_varied_param:

            # The dictionary of kwargs gets updated for each iteration of the varied
            # parameter. Only the varied parameter is changed (all others are fixed).  
            
            kwargs_specific_iter[str(name_of_varied_param)] = specific_varied_value
        
            ### Case 1 (default): Plot only the (reduced) chi-squared values ###
            
            if not plot_sigma_rms:
                
                if without_mag and with_mag:
                    
                    quality_fn_no_mag = compute_a_model_and_save_results(**kwargs_specific_iter, num_dof = num_dof,
                                                without_abs_mag = True, with_abs_mag = False, return_sigma_rms = False,
                                                measured_period_array = measured_period_array,
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_detailed_error_messages = include_detailed_error_messages)                     
    
                    chi_squared_no_mag.append(quality_fn_no_mag)
                    
                    quality_fn_with_mag = compute_a_model_and_save_results(**kwargs_specific_iter, num_dof = num_dof,
                                                without_abs_mag = False, with_abs_mag = True, return_sigma_rms = False,
                                                measured_period_array = measured_period_array,  
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_absolute_magnitude = True, 
                                                Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                                                measured_absolute_magnitude = measured_absolute_magnitude,
                                                measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                                                include_detailed_error_messages = include_detailed_error_messages)
                    
                    chi_squared_with_mag.append(quality_fn_with_mag)
    
                # This block (below) calculates the chi-squared values without absolute magnitudes:
                
                elif without_mag and not with_mag: 
                
                    quality_fn_no_mag = compute_a_model_and_save_results(**kwargs_specific_iter, num_dof = num_dof,
                                                without_abs_mag = True, with_abs_mag = False, return_sigma_rms = False,
                                                measured_period_array = measured_period_array,
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_detailed_error_messages = include_detailed_error_messages)        
    
                    chi_squared_no_mag.append(quality_fn_no_mag)
                    
                # This block calculates the chi-squared values with absolute magnitudes:
                
                elif with_mag and not without_mag:
                
                    quality_fn_with_mag = compute_a_model_and_save_results(**kwargs_specific_iter, num_dof = num_dof,
                                                without_abs_mag = False, with_abs_mag = True, return_sigma_rms = False,
                                                measured_period_array = measured_period_array,  
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_absolute_magnitude = True, 
                                                Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                                                measured_absolute_magnitude = measured_absolute_magnitude,
                                                measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                                                include_detailed_error_messages = include_detailed_error_messages)
                    
                    chi_squared_with_mag.append(quality_fn_with_mag)

            ### Case 2: Plot both the (reduced) chi-squared and sigma rms values ###

            elif plot_sigma_rms:

                if without_mag and with_mag:
                    
                    quality_fn_no_mag, sigma_no_mag = compute_a_model_and_save_results(**kwargs_specific_iter, 
                                                num_dof = num_dof, without_abs_mag = True, with_abs_mag = False, 
                                                return_sigma_rms = True, measured_period_array = measured_period_array,
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_detailed_error_messages = include_detailed_error_messages)                     
                    
                    chi_squared_no_mag.append(quality_fn_no_mag)
                    # print(f"Quality function no mag: {quality_fn_no_mag}")
                    
                    sigma_rms_no_mag.append(sigma_no_mag)
                    # print(f"Sigma rms no mag: {sigma_no_mag}")
                    
                    quality_fn_with_mag, sigma_with_mag = compute_a_model_and_save_results(**kwargs_specific_iter, 
                                                num_dof = num_dof, without_abs_mag = False, with_abs_mag = True, 
                                                return_sigma_rms = True, measured_period_array = measured_period_array,  
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_absolute_magnitude = True, 
                                                Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                                                measured_absolute_magnitude = measured_absolute_magnitude,
                                                measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                                                include_detailed_error_messages = include_detailed_error_messages)
                    
                    # print(f"Quality function with mag: {quality_fn_with_mag}")
                    # print(f"Sigma rms with mag: {sigma_with_mag}")
                    
                    chi_squared_with_mag.append(quality_fn_with_mag)
                    sigma_rms_with_mag.append(sigma_with_mag)

                # This block (below) calculates the chi-squared values without absolute magnitudes:
                
                elif without_mag and not with_mag: 
                
                    quality_fn_no_mag, sigma_no_mag = compute_a_model_and_save_results(**kwargs_specific_iter, 
                                                num_dof = num_dof, without_abs_mag = True, with_abs_mag = False, 
                                                return_sigma_rms = True, measured_period_array = measured_period_array,
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_detailed_error_messages = include_detailed_error_messages)        
    
                    chi_squared_no_mag.append(quality_fn_no_mag)
                    sigma_rms_no_mag.append(sigma_no_mag)
                  
                # This block calculates the chi-squared values with absolute magnitudes:
                
                elif with_mag and not without_mag:
                
                    quality_fn_with_mag, sigma_with_mag = compute_a_model_and_save_results(**kwargs_specific_iter, 
                                                num_dof = num_dof, without_abs_mag = False, with_abs_mag = True, 
                                                return_sigma_rms = True, measured_period_array = measured_period_array,  
                                                measured_period_uncertainty_array = measured_period_uncertainty_array,
                                                include_absolute_magnitude = True, 
                                                Bergeron_filter_if_applicable = Bergeron_filter_if_applicable,
                                                measured_absolute_magnitude = measured_absolute_magnitude,
                                                measured_absolute_magnitude_uncertainty = measured_absolute_magnitude_uncertainty,
                                                include_detailed_error_messages = include_detailed_error_messages)
                    
                    chi_squared_with_mag.append(quality_fn_with_mag)
                    sigma_rms_with_mag.append(sigma_with_mag)
        
        # Convert to arrays (at the very end); see note above regarding the sigma_rms lists.
        
        chi_squared_no_mag = np.array(chi_squared_no_mag)
        chi_squared_with_mag = np.array(chi_squared_with_mag)

        if plot_sigma_rms:
        
            sigma_rms_no_mag = np.array(sigma_rms_no_mag)
            sigma_rms_with_mag = np.array(sigma_rms_with_mag)
            
    ### Create the plots ###
    
    # Label the x-axis (same in all cases) 
    
    x_ax = format_names_properly(name=str(name_of_varied_param))

    # Default: (Reduced) chi-squared plot 

    if not plot_sigma_rms:

        y_ax_chi_sq = format_names_properly(name='red_chi_sq' if num_dof is not None and 
                                            num_dof>0 else 'chi_sq')

        # Case (T,T): Display both sets of chi-squared values on the same plot
        
        if without_mag and with_mag:

            plt.figure()
            plt.scatter(interval_of_varied_param, chi_squared_no_mag, label='without absolute magnitude')
            plt.scatter(interval_of_varied_param, chi_squared_with_mag, label='with absolute magnitude')
            
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_chi_sq)

            plt.title(f"{y_ax_chi_sq} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')
    
            if save_chi_sq_plot:
                
                plt.savefig(full_path_chi_sq, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()
    
        # Case (T,F): Only display the first set of chi-squared values (without absolute magnitude)
        
        elif without_mag and not with_mag:

            plt.figure()
            plt.scatter(interval_of_varied_param, chi_squared_no_mag, label='without absolute magnitude')
    
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_chi_sq)

            plt.title(f"{y_ax_chi_sq} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')
    
            if save_chi_sq_plot:
                
                plt.savefig(full_path_chi_sq, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()
    
        # Case (F,T): Only display the second set of chi-squared values (with absolute magnitude)
        
        elif with_mag and not without_mag:

            plt.figure()
            plt.scatter(interval_of_varied_param, chi_squared_with_mag, label='with absolute magnitude')
    
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_chi_sq)

            plt.title(f"{y_ax_chi_sq} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')
    
            if save_chi_sq_plot:
                
                plt.savefig(full_path_chi_sq, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()
    
    ### Both plots: (reduced) chi-squared and (reduced) sigma rms ###

    elif plot_sigma_rms:

        y_ax_chi_sq = format_names_properly(name='red_chi_sq' if num_dof is not None and
                                            num_dof>0 else 'chi_sq')
        y_ax_sigma_rms = format_names_properly(name='sigma_rms')

        # Same as above:
        
        if without_mag and with_mag:

            # Case (T,T): Display both sets of chi-squared plots (with and without absolute magnitude)

            plt.figure()
            plt.scatter(interval_of_varied_param, chi_squared_no_mag, label='without absolute magnitude')
            plt.scatter(interval_of_varied_param, chi_squared_with_mag, label='with absolute magnitude')
            
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_chi_sq)

            plt.title(f"{y_ax_chi_sq} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')
    
            if save_chi_sq_plot:
                
                plt.savefig(full_path_chi_sq, facecolor='white', transparent=False)

            plt.tight_layout()
            plt.show()

            # Case (T,T): Display both sets of sigma rms plots (with and without absolute magnitude)

            plt.figure()
            plt.scatter(interval_of_varied_param, sigma_rms_no_mag, label='without absolute magnitude')
            plt.scatter(interval_of_varied_param, sigma_rms_with_mag, label='with absolute magnitude')
            
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_sigma_rms)

            plt.title(f"{y_ax_sigma_rms} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else  
                      f"{y_ax_sigma_rms} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_sigma_rms} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')

            if save_sigma_rms_plot:

                plt.savefig(full_path_sigma_rms, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()
        
        elif without_mag and not with_mag:

            # Case (T,F): Only display the first set of chi-squared values (without absolute magnitude)

            plt.figure()
            plt.scatter(interval_of_varied_param, chi_squared_no_mag, label='without absolute magnitude')
    
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_chi_sq)

            plt.title(f"{y_ax_chi_sq} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')
    
            if save_chi_sq_plot:
                
                plt.savefig(full_path_chi_sq, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()

            # Case (T,F): Only display the first set of sigma rms values (without absolute magnitude)

            plt.figure()
            plt.scatter(interval_of_varied_param, sigma_rms_no_mag, label='without absolute magnitude')
    
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_sigma_rms)

            plt.title(f"{y_ax_sigma_rms} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else  
                      f"{y_ax_sigma_rms} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_sigma_rms} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')

            if save_sigma_rms_plot:

                plt.savefig(full_path_sigma_rms, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()

        elif with_mag and not without_mag:

            # Case (F,T): Only display the second set of chi-squared values (with absolute magnitude)

            plt.figure()
            plt.scatter(interval_of_varied_param, chi_squared_with_mag, label='with absolute magnitude')
    
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_chi_sq)

            plt.title(f"{y_ax_chi_sq} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_chi_sq} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')
    
            if save_chi_sq_plot:
                
                plt.savefig(full_path_chi_sq, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()
        
            # Case (F,T): Only display the second set of sigma rms values (with absolute magnitude)

            plt.figure()
            plt.scatter(interval_of_varied_param, sigma_rms_with_mag, label='with absolute magnitude')
    
            plt.xlabel(x_ax)
            plt.ylabel(y_ax_sigma_rms)

            plt.title(f"{y_ax_sigma_rms} vs. {x_ax}\n (1 degree of freedom)" if num_dof==1 else  
                      f"{y_ax_sigma_rms} vs. {x_ax}\n ({num_dof} degrees of freedom)" if num_dof>1 else 
                      f"{y_ax_sigma_rms} vs. {x_ax}", fontsize=12)
            
            plt.legend(loc='best')

            if save_sigma_rms_plot:

                plt.savefig(full_path_sigma_rms, facecolor='white', transparent=False)
            
            plt.tight_layout()
            plt.show()
