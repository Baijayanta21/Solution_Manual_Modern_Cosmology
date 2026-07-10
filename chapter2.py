import numpy as np
from scipy.integrate import quad


def E(a, Omega_m, Omega_r, Omega_l):
    r'''
    Given the scale factor and relevant cosmological parameters, it returns the value(s) of dimensionless Hubble
    parameter :math:`E(a)` which is defined as : 

    .. math::

        E(a) = \sqrt{\Omega_{\Lambda} + \Omega_{m}\, a^{-3} + \Omega_{r}\, a^{-4}}
    
    Parameters
    ----------
    a : float or np.ndarray
        Value of scale factor
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float or np.ndarray
        Dimensionless Hubble parameter value(s).

    '''
    return np.sqrt(Omega_l + Omega_m /(a**3) + Omega_r /(a**4))


def H(a, H_0, Omega_m, Omega_r, Omega_l):
    r'''
    Given the scale factor and relevant cosmological parameters, it returns the value(s) of Hubble
    parameter :math:`H(a)` which is given by : 

    .. math::

        H(a) = H_0\,\sqrt{\Omega_{\Lambda} + \Omega_{m}\, a^{-3} + \Omega_{r}\, a^{-4}}
    
    Parameters
    ----------
    a : float or np.ndarray
        Value of scale factor
    H_0 : float
        Value of Hubble parameter at present in units of km/s/Mpc.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float or np.ndarray
        Hubble parameter value(s) in units of km/s/Mpc.

    '''

    return H_0*E(a, Omega_m, Omega_r, Omega_l)


def cosmic_age_Euc(a, H_0, Omega_m, Omega_r, Omega_l):
    r'''
    Given the scale factor and relevant cosmological parameters, it returns the cosmic age in an Euclidean universe given by :  

    .. math::

        t = \int_{0}^{a}\,\frac{da'}{a'\,H(a')}
    
    Parameters
    ----------
    a : float 
        Value of scale factor at which cosmic age has to be evaluated.
    H_0 : float
        Value of Hubble parameter at present in units of km/s/Mpc.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float or np.ndarray
        cosmic age value in yrs.

    '''

    ai = 0 # lower limit of scale factor
    af = a # upper limit of scale factor
    
    return quad(lambda a: 1/ (a * E(a, Omega_m, Omega_r, Omega_l)), ai, af)[0]/(H_0.to(1 / u.yr))

    # (H0.to(1 / u.yr)) is to convert it to yr units


def conformal_time_Euc(a, H_0, Omega_m, Omega_r, Omega_l):
    r'''
    Given the scale factor and relevant cosmological parameters, it returns the conformal time :math:`\eta(a)` in an Euclidean universe given by :  

    .. math::

        \eta(a) = \int_{0}^{a}\,\frac{da'}{a'^2H(a')}
    
    Parameters
    ----------
    a : float 
        Value of scale factor at which conformal time has to be evaluated.
    H_0 : float
        Value of Hubble parameter at present in units of km/s/Mpc.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float or np.ndarray
        conformal time value in yrs.

    '''

    ai = 0 # lower limit of scale factor
    af = a # upper limit of scale factor
    
    return quad(lambda a: 1/ (a * a * E(a, Omega_m, Omega_r, Omega_l)), ai, af)[0]/(H_0.to(1 / u.yr))

    # (H0.to(1 / u.yr)) is to convert it to yr units
    

def comov_distance_Euc(z, H_0, Omega_m, Omega_r, Omega_l):
    r'''
    Given the redshift and relevant cosmological parameters, it returns the value of comoving distance
    in an Euclidean universe :math:`\chi(z)` which is given by : 

    .. math::

        \chi^{\rm Euc}(z) = \int_{a(z)}^{1}\,\frac{da'}{a'^2H(a')}, \quad \text{where}, \quad a(z) = \frac{1}{1+z}\\[1em]
    
    Parameters
    ----------
    z : float
        Redshift factor value.
    H_0 : float
        Value of Hubble parameter at present in units of km/s/Mpc.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float 
        Comoving distance in units of Mpc.

    '''

    ai = 1/(1+z) # lower limit of scale factor
    af = 1       # upper limit of scale factor
    
    return quad(lambda a: 1/ (a * a * E(a, Omega_m, Omega_r, Omega_l)) , ai, af)[0]/((H_0 / c).to(1 / u.Mpc))

    # ((H_0 / c).to(1 / u.Mpc)) is to due to the natural units convention we followed from the start


def d_A_Euc(z, H_0, Omega_m, Omega_r, Omega_l):
    r'''
    Given the redshift and relevant cosmological parameters, it returns the value of angular diameter distance
    in an Euclidean universe :math:`d_{A}^{\rm Euc}(z)` which is given by : 

    .. math::

        d_{A}^{\rm Euc}(z) = \frac{\chi^{\rm Euc}(z)}{1+z} 
    
    Parameters
    ----------
    z : float
        Redshift factor value.
    H_0 : float
        Value of Hubble parameter at present in units of km/s/Mpc.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float 
        Angular diameter distance in units of Mpc.

    '''

    return comov_distance_Euc(z, H_0, Omega_m, Omega_r, Omega_l)/(1+z)
    

def d_L_Euc(z, H_0, Omega_m, Omega_r, Omega_l):
    r'''
    Given the redshift and relevant cosmological parameters, it returns the value of luminosity distance
    in an Euclidean universe :math:`d_{L}^{\rm Euc}(z)` which is given by : 

    .. math::

        d_{L}^{\rm Euc}(z) = ({1+z})\cdot \chi^{\rm Euc}(z)
    
    Parameters
    ----------
    z : float
        Redshift factor value.
    H_0 : float
        Value of Hubble parameter at present in units of km/s/Mpc.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float 
        Luminosity distance in units of Mpc.

    '''

    return comov_distance_Euc(z, H_0, Omega_m, Omega_r, Omega_l)*(1+z)


import astropy.units as u
from astropy.constants import c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
plt.style.use('/home/cts23ph/SP_FINAL/mplstyle')




# define Hubble paramater, Use best fit values
h = 0.6770
H_0 = 100 * h * u.km / u.s / u.Mpc


# #### $\text{Problem 2.5}$

print(cosmic_age_Euc(a = 2.349e-9 , H_0 = H_0, Omega_m = 0.3106, Omega_r = 4.15e-5/(h**2), Omega_l = 0).to(u.s))
print(cosmic_age_Euc(a = 9.396e-4 , H_0 = H_0, Omega_m = 0.3106, Omega_r = 4.15e-5/(h**2), Omega_l = 0 ).to(u.yr))


# #### $\text{Problem 2.6}$

print(conformal_time_Euc(a = 1/1101, H_0 = H_0, Omega_m = 0.3106, Omega_r = 4.15e-5/(h**2), Omega_l = 0))


# #### $\text{Problem 2.7}$

H_0 = 100 * u.km / u.s / u.Mpc # without h form
l   = 5 * u.kpc                # object size 

Omega_m = 0.3106
Omega_l = 0.6894


print("-"*15 + "  Matter Dominated " + "-"*15)

print(f'z = {0.1} : comoving distance  = {comov_distance_Euc(z = 0.1, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0):.4f}' )
print(f'z = {1.0} : comoving distance  = {comov_distance_Euc(z = 1.0, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0):.4f}' )

print(f'z = {0.1} : angular extent     = {(l / d_A_Euc(z = 0.1, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0) * u.rad).to(u.arcsec):.4f}' )
print(f'z = {1.0} : angular extent     = {(l / d_A_Euc(z = 1.0, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0) * u.rad).to(u.arcsec):.4f}' )

print("-"*15 + " Fiducial Cosmology " + "-"*14)

print(f'z = {0.1} : comoving distance  = {comov_distance_Euc(z = 0.1, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l):.4f}' )
print(f'z = {1.0} : comoving distance  = {comov_distance_Euc(z = 1.0, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l):.4f}' )

print(f'z = {0.1} : angular extent     = {(l / d_A_Euc(z = 0.1, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l) * u.rad).to(u.arcsec):.4f}' )
print(f'z = {1.0} : angular extent     = {(l / d_A_Euc(z = 1.0, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l) * u.rad).to(u.arcsec):.4f}' )

print("-"*49)


# #### $\text{Problem 2.10}$


# redshift array 
z = np.logspace(-2, 0, 50)
H_0 = 100 * u.km / u.s / u.Mpc

chi = np.array([comov_distance_Euc(zi, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0).value for zi in z])
d_A = np.array([d_A_Euc(zi, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0).value for zi in z])
d_L = np.array([d_L_Euc(zi, H_0 = H_0, Omega_m = 1, Omega_r = 0, Omega_l = 0).value for zi in z])


plt.plot(z, chi, 'k'  , lw = 1.5, label = r'$\chi$')
plt.plot(z, d_A, 'r--', lw = 1.5, label = r'$d_A$')
plt.plot(z, d_L, 'g-.', lw = 1.5, label = r'$d_L$')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$z$')

# Display numbers normally (no scientific notation)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

plt.ylabel('Distance ' + '[$h^{-1}$' +'Mpc]')
plt.title(r'$ \Omega_{m} = 1$ ', fontsize = 14)
plt.legend(fontsize = 15)

plt.savefig("distance_md.pdf", dpi = 700, transparent = True, bbox_inches = "tight", pad_inches = 0.1)
plt.show()


plt.plot(z, 5*np.log10(1e5*d_L), ls = '-', marker = 'D', ms = 3, mfc = 'white', mec = 'k')

plt.xscale('log')
plt.xlabel('$z$')

# Display numbers normally (no scientific notation)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

plt.ylabel('Distance Modulus\,' + r'$(\mu)$')
plt.title(r'$ \Omega_{m} = 1$ ', fontsize = 14)

plt.savefig("distance_modulus_md.pdf", dpi = 700, transparent = True, bbox_inches = "tight", pad_inches = 0.1)
plt.show()


# =================================================================================================================

H_0 = 100 * h * u.km / u.s / u.Mpc

Omega_m = 0.3106
Omega_l = 0.6894

chi = np.array([comov_distance_Euc(zi, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l).value for zi in z])
d_A = np.array([d_A_Euc(zi, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l).value for zi in z])
d_L = np.array([d_L_Euc(zi, H_0 = H_0, Omega_m = Omega_m, Omega_r = 0, Omega_l = Omega_l).value for zi in z])


# In[33]:


plt.plot(z, chi, 'k'  , lw = 1.5, label = r'$\chi$')
plt.plot(z, d_A, 'r--', lw = 1.5, label = r'$d_A$')
plt.plot(z, d_L, 'g-.', lw = 1.5, label = r'$d_L$')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$z$')

# Display numbers normally (no scientific notation)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

plt.ylabel('Distance ' + '[Mpc]')

plt.title(r'$ h = 0.677,\,\Omega_{m} = 0.3106,\,\Omega_{\Lambda} = 0.6894$', fontsize = 14)
plt.legend(fontsize = 15)

plt.savefig("distance_lcdm.pdf", dpi = 700, transparent = True, bbox_inches = "tight", pad_inches = 0.1)
plt.show()


plt.plot(z, 5*np.log10(1e5*d_L), ls = '-', marker = 'D', ms = 3, mfc = 'white', mec = 'k')

plt.xscale('log')
plt.xlabel('$z$')

# Display numbers normally (no scientific notation)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

plt.ylabel('Distance Modulus\,' + r'$(\mu)$')
plt.title(r'$ h = 0.677,\,\Omega_{m} = 0.3106,\,\Omega_{\Lambda} = 0.6894$', fontsize = 14)

plt.savefig("distance_modulus_lcdm.pdf", dpi = 700, transparent = True, bbox_inches = "tight", pad_inches = 0.1)
plt.show()


# #### $\text{Problem 2.14}$

def rho_m(a, Omega_m):
    r'''
    Given the scale factor :math:`a` and density parameter of matter (:math:`\Omega_{m}`) at present, it returns 
    the matter energy density :math:`\rho_m(a)` by : 

    .. math::

        \rho_m(a) = \Omega_{m}\, \rho_{\rm cr}\, a^{-3}
        
    Parameters
    ----------
    a : float 
        Value of scale factor at which cosmic age has to be evaluated.
    Omega_m : float
        Density parameter :math:`\Omega_{m}` corresponding to matter at present.

    Returns
    -------
    float or np.ndarray
        Matter energy density.

    '''

    return Omega_m/a**3 # in units of critical density


def rho_r(a, Omega_r):
    r'''
    Given the scale factor :math:`a` and density parameter of radiation (:math:`\Omega_{r}`) at present, it returns 
    the radiation energy density :math:`\rho_r(a)` by : 

    .. math::

        \rho_r(a) = \Omega_{r}\, \rho_{\rm cr}\, a^{-4}
        
    Parameters
    ----------
    a : float 
        Value of scale factor at which cosmic age has to be evaluated.
    Omega_r : float
        Density parameter :math:`\Omega_{r}` corresponding to radiation at present.

    Returns
    -------
    float or np.ndarray
        Radiation energy density.

    '''

    return Omega_r/a**4 # in units of critical density
    
def rho_l(a, Omega_l):
    r'''
    Given the scale factor :math:`a` and density parameter of dark energy (:math:`\Omega_{\Lambda}`) at present, it returns 
    the dark energy density :math:`\rho_\Lambda(a)` by : 

    .. math::

        \rho_\Lambda(a) = \Omega_{_\Lambda}\, \rho_{\rm cr}
        
    Parameters
    ----------
    a : float 
        Value of scale factor at which cosmic age has to be evaluated.
    Omega_l : float
        Density parameter :math:`\Omega_{\Lambda}` corresponding to dark energy at present.

    Returns
    -------
    float or np.ndarray
        Dark energy density.

    '''

    return Omega_l*np.ones_like(a) # in units of critical density


a = np.logspace(-5, 0, num = 50)   # scale factor

plt.plot(a, rho_l(a, Omega_l = 0.6894),       'g-.', lw = 1.25, label = 'cosmol. const.') # dark      energy density
plt.plot(a, rho_m(a, Omega_m = 0.3106),       'k'  , lw = 1.25, label = 'matter')         # matter    energy density
plt.plot(a, rho_r(a, Omega_r = 4.15e-5/h**2), 'r--', lw = 1.25, label = 'radiation')      # radiation energy density

plt.xlabel('$a$')
plt.ylabel(r'$\rho_s/\rho_{\rm cr}$')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize = 11, bbox_to_anchor = (.9,.95))

a_l = (Omega_m/Omega_l)**(1/3)

a_eq = (4.15e-5/h**2)/0.3106

# annotate the equality scale factors
plt.annotate(r'$a_{\Lambda}$',xy = (a_l, Omega_l), xytext=(0, 50), textcoords = 'offset points', arrowprops = dict(arrowstyle = '-|>'), fontsize = 15, ha = 'center', va = 'center')
plt.annotate(r'$a_{\rm eq}$', xy = (a_eq, rho_m(a_eq, Omega_m = 0.3106)), xytext = (0, 50), textcoords = 'offset points', arrowprops = dict(arrowstyle = '-|>'), fontsize = 15 , ha = 'center', va = 'center')

plt.savefig("density.pdf", dpi = 700, transparent = True, bbox_inches = "tight", pad_inches = 0.1)
plt.show()

