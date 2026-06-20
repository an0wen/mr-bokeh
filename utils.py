import numpy as np
import pandas as pd
import astropy.units as u
from numpy.random import *
from astropy import units

def load_planet_data(fname_base, fname_suffix='_CLEANED', fname_extension='.csv', data_dir='data/nea_cleaned/'):

    # Load and down-select the data
    fname = fname_base + fname_suffix + fname_extension
    X = pd.read_csv(data_dir + fname)

    # Add column for a symmetric Gaussian error bar that's the average of the lower and upper bars
    X['pl_bmasseerr'] = np.mean(np.abs(X[['pl_bmasseerr1', 'pl_bmasseerr2']]), axis=1)
    X['pl_radeerr']  = np.mean(np.abs(X[['pl_radeerr1', 'pl_radeerr2']]), axis=1)

    X['log_pl_bmasse'] = np.log(X['pl_bmasse'].values.astype(float))
    X['log_pl_rade'] = np.log(X['pl_rade'].values.astype(float))

    X['log_pl_bmasseerr'] = X['pl_bmasseerr'].values / X['pl_bmasse'].values
    X['log_pl_radeerr']   = X['pl_radeerr'].values / X['pl_rade'].values
    X['log_weights'] = 1 / (np.square(X['log_pl_bmasseerr']) + np.square(X['log_pl_radeerr']))

    return X

def rotate(p, origin=(0, 0), degrees=0):
    '''
    https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    '''
    
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def generate_monte_carlo_dataset(X, xkey='pl_bmasse', xerrkey='pl_bmasseerr', ykey='pl_rade', yerrkey='pl_radeerr'):
    
    X_mc = pd.DataFrame(columns=[xkey, xerrkey, ykey, yerrkey])
    X_mc[xerrkey] = X[xerrkey].copy()
    X_mc[yerrkey] = X[yerrkey].copy()
    
    for i in range(len(X)):
        
        X_mc.loc[i, xkey] = np.random.normal(X.loc[i, xkey], X.loc[i, xerrkey])
        X_mc.loc[i, ykey]   = np.random.normal(X.loc[i, ykey], X.loc[i, yerrkey])
        
    return X_mc

def get_teq_from_sinc(sinc, bond_albedo=0):
    '''
    sinc in units of earth flux
    Returns in units of K.
    '''
    TEFF_SOL = 5772 # K.
    return TEFF_SOL * np.power(sinc, 1/4) * 1 / np.sqrt(u.AU.to(u.Rsun, 1) * 2) * (1 - bond_albedo)**(0.25)

def get_semimajor_axis(period, m_star):
    '''
    Return a in units of AU.
    '''
    period /= 365.25 # Convert JD to years
    return np.cbrt(np.square(period) * m_star)

def get_luminosity(teff, r_star):
    '''
    Return L in units of L_sun
    '''
    TEFF_SOL = 5772 # K.
    return np.square(r_star) * np.power(teff/TEFF_SOL, 4)

def get_sinc(pl_period, teff, m_star, r_star):
    '''
    Return insolation flux in units of S_earth
    '''
    luminosity = get_luminosity(teff, r_star)
    a = get_semimajor_axis(pl_period, m_star)
    return luminosity / np.square(a)

def get_teq(pl_period, teff, m_star, r_star, bond_albedo=0):
    '''
    Return planet equilibrium temperature in units of Kelvin
    '''
    a = get_semimajor_axis(pl_period, m_star)
    a_sun = (a.values * u.AU).to(u.R_sun).value
    return teff * (1 - bond_albedo)**(0.25) * np.sqrt(r_star / (2 * a_sun))

def get_dens(pl_masse, pl_rade):
    '''
    Returns planet bulk density in units of g/cm^3. Assuming planet mass and radius given in Earth units.
    '''
    return (u.M_earth / u.R_earth**3).to(u.g / u.cm**3, (pl_masse / (4/3 * np.pi * pl_rade**3)).values)

# From https://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text, text_positions, axis,txt_width,txt_height):
    for x,y,text,t in zip(x_data, y_data, text, text_positions):
        axis.text(x - txt_width, 1.01*t, text,rotation=0, color='blue')
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                       head_width=txt_width, head_length=txt_height*0.5, 
                       zorder=0,length_includes_head=True)

def get_tsm(pl_rade, pl_masse, pl_aor, rstar, teff, jmag):
    '''
    Calculate TSM from derived chains.
    '''
    pl_rade_med = np.median(pl_rade) # Determine the scale factor using the median of the planet's radius measurement.
    scale_factor = None
    if pl_rade_med < 1.5:
        scale_factor = 0.190
    elif pl_rade_med >= 1.5 and pl_rade_med < 2.75:
        scale_factor = 1.26
    elif pl_rade_med >= 2.75 and pl_rade_med < 4.0:
        scale_factor = 1.28
    elif pl_rade_med >= 4.0 and pl_rade_med < 10:
        scale_factor = 1.15
    else:
        scale_factor = -1 # Planet too large
    teq = teff * (np.sqrt(1/pl_aor)*(0.25**0.25))

    numerator = scale_factor * pl_rade**3 * teq * 10**(-1 * jmag / 5)
    denominator = pl_masse * rstar**2

    return numerator / denominator

def get_aor(a_samples, rstar_samples):
    return (a_samples * units.AU).to(units.R_sun).value / rstar_samples