# Code adapted from Gavin, H.P. (2020) The Levenberg-Marquardt method for 
# nonlinear least squares curve-fitting problems.
# https://people.duke.edu/~hpgavin/ce281/lm.pdf
    
import numpy as np
import matplotlib.pyplot as plt
import levenberg_marquardt as LM

def make_noisy_test_data(p_true,Npnt,msmnt_err):
    """
    
    Make noisy test data.

    Parameters
    ----------
    p_true      : true fitted parameters values (n x 1), must be 2D array
    Npnt        : number of data points (m = Npnt)
    msmnt_err   : amount of noise added to model data

    Returns
    -------
    x           : x-values of test data (m x 1), must be 2D array
    y           : y-values of test data (m x 1), must be 2D array

    """
    
    x = np.array(range(Npnt)).T
    y_true = LM.lm_func(x,p_true)
    
    # add random measurement errors
    rng = np.random.default_rng()
    y = y_true + msmnt_err*rng.random((Npnt)) 
    
    return x,y

def main(x,y,p_init):
    """
    
    Main function for performing Levenberg-Marquardt curve fitting.

    Parameters
    ----------
    x           : x-values of input data (m x 1), must be 2D array
    y           : y-values of input data (m x 1), must be 2D array
    p_init      : initial guess of parameters values (n x 1), must be 2D array
                  n = 4 in this example

    Returns
    -------
    p       : least-squares optimal estimate of the parameter values
    Chi_sq  : reduced Chi squared error criteria - should be close to 1
    sigma_p : asymptotic standard error of the parameters
    sigma_y : asymptotic standard error of the curve-fit
    corr    : correlation matrix of the parameters
    R_sq    : R-squared cofficient of multiple determination  
    cvg_hst : convergence history (col 1: function calls, col 2: reduced chi-sq,
              col 3 through n: parameter values). Row number corresponds to
              iteration number.

    """
    
    # close all plots
    plt.close('all')
    
    # minimize sum of weighted squared residuals with L-M least squares analysis
    p_fit,Chi_sq,sigma_p,sigma_y,corr,R_sq,cvg_hst = LM.lm(p_init,x,y)
    
    # plot results of L-M least squares analysis
    LM.make_lm_plots(x, y, cvg_hst)
    
    return p_fit,Chi_sq,sigma_p,sigma_y,corr,R_sq,cvg_hst
    
if __name__ == '__main__':
    
    # flag for making noisy test data
    make_test_data = True
    
    # make test data with noise
    if make_test_data:
        # define true fitted parameters for testing (must be 2D array)
        p_true = np.array([[6,20,1,5]]).T
        # define initial guess of parameters (must be 2D array)
        p_init = np.array([[10,50,5,5.7]]).T
        # number of data points (x-values will range from 0 to 99)
        Npnt = 100 
        # adding noise to input data to simulate artificial measurements
        msmnt_err = 0.5 
        
        x,y = make_noisy_test_data(p_true,Npnt,msmnt_err)
        
    p_fit,Chi_sq,sigma_p,sigma_y,corr,R_sq,cvg_hst = main(x,y,p_init)
