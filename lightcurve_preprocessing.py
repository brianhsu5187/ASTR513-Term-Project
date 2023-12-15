#!/usr/bin/env python
# coding: utf-8
# %%
from astropy.table import Table, vstack
from matplotlib import pyplot as plt
import numpy as np
import random
from lightcurve_fitting.lightcurve import LC
from dust_extinction.parameter_averages import G23
from astropy import units as u
from scipy.stats import wasserstein_distance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, PairwiseKernel
from scipy.optimize import minimize
import speclite.filters
from astropy import units as u
from astropy.io import ascii
from astropy.cosmology import Planck18 as cosmo
import datetime
import os
import logging

from glob import glob
from itertools import chain
import argparse


# %%
def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

all_lc = glob('LC_data/*')
transient_class = dict(SNIa=90,SNIa91bg=67,SNIax=52,SNII=42,SNIbc=62,SLSNI=95,TDE=15,KN=64,AGN=88,
                       ILOT=992,CaRT=993,PISN=994)
class_frac = {clas:len(glob('LC_data/'+clas+'*'))/len(all_lc) for clas in list(transient_class.keys())}
# train_list = [random.sample(glob('LC_data/'+clas+'_id*.dat'),int(0.01*len(all_lc)*frac)) for clas, frac 
#               in list(class_frac.items())]
# train_list = flatten_chain(train_list)

feat_data = np.load('features/feat_train.npz', allow_pickle=True)
ids = feat_data['ids']
train_list = ['LC_data/'+idd+'.dat' for idd in ids]
test_list = list(set(all_lc)-set(train_list))


# %%


lsst = speclite.filters.load_filters(f'lsst2016-*')

filters = 'ugrizy'
colors = ['darkblue','green','red','orange','purple','black']
max_band = len(filters)

data = {f: speclite.filters.load_filters(f'lsst2016-{f}')[0] for f in filters}
wavelength = np.linspace(3000, 11000, num=10000)*u.AA

normalized = {f: data[f](wavelength)/data[f](wavelength).sum() for f in filters}

def distance_between_filters(filter1, filter2):
    return wasserstein_distance(u_values=wavelength.value, v_values=wavelength.value,
                                u_weights=normalized[filter1], v_weights=normalized[filter2])

distance_matrix = np.array([[distance_between_filters(col, row) for col in filters] for row in filters])
distance_matrix /= np.average(distance_matrix)


# %%


def run_gp(Xt, Xf, Xfl, Xfle):

    mag_var = np.var(Xfl)

    def metric(x1, x2, p):
        band1 = (x1[1].astype(int))
        band2 = (x2[1].astype(int))
        time_distance = x2[0] - x1[0]
        photometric_distance = distance_matrix[band1, band2]

        return (
            mag_var*np.exp(-photometric_distance**2/(2*p[0]**2) - time_distance**2/(2*p[1]**2))
            )
        
    def fit_gp(X, y, p):
        cur_metric = lambda x1, x2, gamma: metric(x1, x2, p)

        kernel = PairwiseKernel(metric=cur_metric)
        gp = GaussianProcessRegressor(kernel=kernel,
                                    alpha=y[:, 1]**2,
                                    normalize_y=False)
        gp.fit(X, y[:, 0])
        return gp

    cX = np.stack(
        (Xt, Xf), axis=1).astype(np.float64)
    cy = np.stack(
        (Xfl, Xfle), axis=1).astype(np.float64)

    def try_lt_lp(p):
        summed_log_like = 0.0
        gp = fit_gp(cX, cy, p)
        summed_log_like += gp.log_marginal_likelihood()

        return -summed_log_like
    
    
    res = minimize(
        lambda x: try_lt_lp(np.exp(x)),
        [0.0, 7.0], bounds=[(0, Xf.max()-Xf.min()),(0,np.diff(Xt).max())])
    best_lengths = np.exp(
        res.x
    )

    gp = fit_gp(cX, cy, best_lengths)
    all_mus = []
    all_stds = []
    for band in range(max_band):
        times = np.linspace(-100,200,300)
        test_data = np.array([[it, band] for it in times])
        mu, std = gp.predict(test_data, return_std=True)
        all_mus.append(mu)
        all_stds.append(std**2) #this is var, hacky
    
    all_stds = np.array(all_stds)
    all_mus = np.array(all_mus)
    return all_mus, all_stds, gp, times


# %%


def preprocess_LC(train_list, meta_file='meta_all.csv'):
    """
    Read in LC files and convert to LC object

    Parameters
    ----------
    lcs_table : astropy.table.Table
        List of LC file names, to be read in.
    obj_names : list
        List of SNe names, should be same length as input_files
    style : string
        Style of LC files. Assumes SNANA

    Returns
    -------
    lcs : list
        list of Light Curve objects

    Examples
    --------
    """
    LC_list = []
    meta_data = Table.read(meta_file,format='ascii')
    filesize = len(train_list)
    for n, event in enumerate(train_list):
        lc = Table.read(event,format='ascii')
        lc = lc[lc['detected_bool']==1]
        t = np.asarray(lc['mjd'],dtype=float)
        f = np.asarray(lc['flux'],dtype=float)
        err = np.asarray(lc['flux_err'],dtype=float)
        filts = np.asarray(lc['passband'],dtype=int)
        sn_name = event.split('/')[-1][:-4]
        my_lc = LightCurve(sn_name, t, f, err, filts)

        meta = meta_data[meta_data['object_id']==lc['object_id'][0]]

        my_lc.add_LC_info(zpt=27.5, mwebv=float(meta['mwebv']), redshift=float(meta['hostgal_photoz']), 
                          redshift_err = float(meta['hostgal_photoz_err']), lim_mag=26,
                          obj_type=sn_name.split('_')[0])
        my_lc.get_abs_mags()
        if np.inf in my_lc.abs_mags:
            print(f'Object {sn_name} not processed: invalid absolute magnitude found.')
            continue
        my_lc.sort_lc()
        if my_lc.times.size < 1:
            print(f'Object {sn_name} not processed: not enough data points.')
            continue

        pmjd = my_lc.find_peak(float(meta['true_peakmjd']))
        my_lc.shift_lc(pmjd)
        my_lc.correct_time_dilation()
        my_lc.cut_lc()
        if my_lc.times.size < 3:
            print(f'Object {sn_name} not processed: not enough data points.')
            continue
        my_lc.make_dense_LC();
        if n%10 != 0:
            print(f'Object {sn_name} processed.')
        else:
            print(f'Object {sn_name} processed. {100*n/filesize}% completed')
        LC_list.append(my_lc)
        
    return LC_list


# %%


class LightCurve(object):
    """Light Curve class
    """
    def __init__(self, name, times, fluxes, flux_errs, filters,
                 zpt=0, mwebv=0, redshift=None, redshift_err=None,
                 lim_mag=None, obj_type=None):

        self.name = name
        self.times = times
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.filters = filters
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.redshift_err = redshift_err
        self.lim_mag = lim_mag
        self.obj_type = obj_type

        self.abs_mags = None
        self.abs_mags_err = None
        self.abs_lim_mag = None

    def sort_lc(self):
        gind = np.argsort(self.times)
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]

    def find_peak(self, tpeak_guess):
        gind = np.where((np.abs(self.times-tpeak_guess) < 1000.0) &
                        (self.fluxes/self.flux_errs > 3.0))
        if len(gind[0]) == 0:
            gind = np.where((np.abs(self.times - tpeak_guess) < 1000.0))
        if len(gind[0]) == 0:
            tpeak = tpeak_guess
            return tpeak
        if self.abs_mags is not None:
            tpeak = self.times[gind][np.argmin(self.abs_mags[gind])]
        return tpeak

    def cut_lc(self, limit_before=100, limit_after=200):
        gind = np.where((self.times > -limit_before) &
                        (self.times < limit_after))
        self.times = self.times[gind]
        self.fluxes = self.fluxes[gind]
        self.flux_errs = self.flux_errs[gind]
        self.filters = self.filters[gind]
        if self.abs_mags is not None:
            self.abs_mags = self.abs_mags[gind]
            self.abs_mags_err = self.abs_mags_err[gind]

    def shift_lc(self, t0=0):
        self.times = self.times - t0

    def correct_time_dilation(self):
        self.times = self.times / (1.+self.redshift)

    def add_LC_info(self, zpt=27.5, mwebv=0.0, redshift=0.0,redshift_err=0.0,
                    lim_mag=25.0, obj_type='-'):
        self.zpt = zpt
        self.mwebv = mwebv
        self.redshift = redshift
        self.redshift_err = redshift_err
        self.lim_mag = lim_mag
        self.obj_type = obj_type

    def get_abs_mags(self, replace_nondetections=True, mag_err_fill=1.0):
        """
        Convert flux into absolute magnitude

        Parameters
        ----------
        replace_nondetections : bool
            Replace nondetections with limiting mag.

        Returns
        -------
        self.abs_mags : list
            Absolute magnitudes

        Examples
        --------
        """
        lsst_filters = {'0':3740., '1':4870., '2':6250., '3':7700., '4':8900., '5':10845.}
        ext = G23(Rv=3.1)
        reddening = -2.5 * np.log10(ext.extinguish([lsst_filters[str(filt)] for filt 
                                                    in self.filters.astype(int)] * u.AA, 
                                                    Ebv=self.mwebv))
        k_correction = 2.5 * np.log10(1.+self.redshift)
        dist = cosmo.luminosity_distance([self.redshift]).value[0]  # returns dist in Mpc

        self.abs_mags = -2.5 * np.log10(self.fluxes) + self.zpt - 5. *             np.log10(dist*1e6/10.0) + k_correction - reddening
        self.abs_mags_err = np.abs((2.5/np.log(10))*(self.flux_errs/self.fluxes))

        if replace_nondetections:
            abs_lim_mag = self.lim_mag - 5.0 * np.log10(dist * 1e6 / 10.0) +                             k_correction
            gind = np.where((np.isnan(self.abs_mags)) |
                            np.isinf(self.abs_mags) |
                            np.isnan(self.abs_mags_err) |
                            np.isinf(self.abs_mags_err) |
                            (self.abs_mags > self.lim_mag))

            self.abs_mags[gind] = abs_lim_mag
            self.abs_mags_err[gind] = mag_err_fill
        self.abs_lim_mag = abs_lim_mag

        return self.abs_mags, self.abs_mags_err

    def make_dense_LC(self, nfilts=6):
        gp_mags = self.abs_mags - self.abs_lim_mag
        dense_fluxes = np.zeros((len(self.times), nfilts))
        dense_errs = np.zeros((len(self.times), nfilts))
        stacked_data = np.vstack([self.times, self.filters]).T
        x_pred = np.zeros((len(self.times)*nfilts, 2))
        print(self.name)

        pred, pred_var, gp, times = run_gp(self.times, self.filters, gp_mags, self.abs_mags_err)
        pred = pred.T
        pred_var = pred_var.T
        self.gp = [1,2,3]

        dense_fluxes = pred + self.abs_lim_mag
        dense_errs = np.sqrt(pred_var)

        self.dense_lc = np.dstack((dense_fluxes, dense_errs))
        self.dense_times = times

        self.gp_mags = gp_mags
        return gp, gp_mags


# %%


def save_lcs(lc_list, output_dir='/Users/brianhsu/ASTR513/ASTR513-Term-Project/preprocessed_lc/', 
             file_suffix = '', filename='lcs.npz'):
    """
    Save light curves as a lightcurve object

    Parameters
    ----------
    lc_list : list
        list of light curve files
    output_dir : Output directory of light curve file
    file_suffix : str
        String to append to file name
    """
    now = datetime.datetime.now()
    date = str(now.strftime("%Y-%m-%d"))
#     file_name = 'lcs_' + date + '_'+file_suffix+'.npz'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_dir[-1] != '/':
        output_dir += '/'

    output_file = output_dir + filename
#     np.savez(output_file, lcs=lc_list)
    # Also easy save to latest
    np.savez(output_file, lcs=lc_list)

    logging.info(f'Saved to {output_file}')


# %%
parser = argparse.ArgumentParser()

parser.add_argument('--num', type=int, help='Light curve file number')
parser.add_argument('--outdir', type=str, help='Output directory',
                    default='/groups/kdalexander/bhsu/ASTR513/preprocessed_lc/')
args = parser.parse_args()

# %%
j = args.num
test = preprocess_LC(test_list[25000*(j-1):25000*j])
filename = 'test_lcs_'+str(j)+'.npz'
save_lcs(test,filename=filename,output_dir=args.outdir)
