from scipy.integrate import quad
import numpy as np



def chi_fiducial(z):
    H_0 = 100*1000/299792458
    omega_m = 0.3106
    omega_lambda = 0.6894

    at = 1/(1+z)
    
    def dcomov(a):
        return 1/np.sqrt(omega_m*a+omega_lambda*a**4)

    return quad(dcomov, at, 1)[0]/H_0
