import numpy as np
from scipy import stats
import argparse
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
import datetime
import os
from glob import glob
from itertools import chain
from astropy.table import Table

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def prep_input(input_lc_file, new_t_max=100.0, filler_err=1.0,
               save=False, load=False, outdir=None, prep_file=None):
    """
    Prep input file for fitting

    Parameters
    ----------
    input_lc_file : str
        True flux values
    new_t_max : float
        Predicted flux values
    filler_err : float
        Predicted flux values
    save : bool
        Predicted flux values
    load : bool
        Predicted flux values
    outdir : str
        Predicted flux values
    prep_file : str
        Predicted flux values

    Returns
    -------
    sequence : numpy.ndarray
        Array LC flux times, values and errors
    outseq : numpy.ndarray
        An array of LC flux values and limiting magnitudes
    ids : numpy.ndarray
        Array of SN names
    sequence_len : float
        Maximum length of LC values
    nfilts : int
        Number of filters in LC files
    """
    IIn_ids = Table.read('plasticc_modelpar_042_SNIIn.csv',format='ascii')['object_id'].value
    lightcurves = np.load(input_lc_file, allow_pickle=True)['lcs']
    names = np.array([lightcurve.name.split('_')[0] for lightcurve in lightcurves])
    ids = np.array([lightcurve.name.split('_')[-1][2:] for lightcurve in lightcurves])
    names[np.nonzero(np.in1d(ids,IIn_ids))[0]]='SNIIn'
    test = []
    for sn in sorted(set(names)):
        inds = np.where(names==sn)[0]
        times = [len(lightcurve.times) for lightcurve in lightcurves[inds]]
        argmax = np.argmax(times)
        test.append(lightcurves[inds][argmax])
        
    names = np.array([lightcurve.name.split('_')[0] for lightcurve in test])
    lengths = flatten_chain([np.arange(1,301,1) for i in range(len(test))])
    ids = flatten_chain([[lightcurve.name for i in range(300)]for lightcurve in test])
    sequence_len = np.max(lengths)
    nfilts = 6
    nfiltsp1 = nfilts+1
    n_lcs = len(test)*300
    sequence = np.zeros((n_lcs, sequence_len, nfilts*2+1))

    lightcurves = flatten_chain([[lightcurve for i in range(300)]for lightcurve in test])
    lms = []
    for i, (lightcurve, time) in enumerate(zip(lightcurves,lengths)):
        sequence[i, 0:lengths[i], 0] = lightcurve.dense_times[:time]
        sequence[i, 0:lengths[i], 1:nfiltsp1] = lightcurve.dense_lc[:, :, 0][:time]
        sequence[i, 0:lengths[i], nfiltsp1:] = lightcurve.dense_lc[:, :, 1][:time] + 0.01
        sequence[i, lengths[i]:, 0] = 200+100
        sequence[i, lengths[i]:, 1:nfiltsp1] = lightcurve.abs_lim_mag
        sequence[i, lengths[i]:, nfiltsp1:] = 1
        lms.append(lightcurve.abs_lim_mag)

    # Flip because who needs negative magnitudes
    sequence[:, :, 1:nfiltsp1] = -1.0 * sequence[:, :, 1:nfiltsp1]

    if load:
        prep_data = np.load(prep_file)
        bandmin = prep_data['bandmin']
        bandmax = prep_data['bandmax']
    else:
        bandmin = np.min(sequence[:, :, 1:nfiltsp1])
        bandmax = np.max(sequence[:, :, 1:nfiltsp1])

    sequence[:, :, 1:nfiltsp1] = (sequence[:, :, 1:nfiltsp1] - bandmin)         / (bandmax - bandmin)

    new_lms = np.reshape(np.repeat(lms, sequence_len), (len(lms), -1))

    outseq = np.reshape(sequence[:, :, 0], (len(sequence), sequence_len, 1)) * 1.0
    outseq = np.dstack((outseq, new_lms))
    if save:
        model_prep_file = outdir+'prep_'+date+'.npz'
        np.savez(model_prep_file, bandmin=bandmin, bandmax=bandmax)
        model_prep_file = outdir+'prep.npz'
        np.savez(model_prep_file, bandmin=bandmin, bandmax=bandmax)
    return sequence, outseq, ids, sequence_len, nfilts

def get_decoder(model, encodingN):
    encoded_input = Input(shape=(None, (encodingN+2)))
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer3(decoder_layer2(encoded_input)))
    return decoder


def get_decodings(decoder, encoder, sequence, lms, encodingN, sequence_len,
                nfilts, ids, plot=True):
    if plot:
        for i in np.arange(len(sequence)):
            seq = np.reshape(sequence[i, :, :], (1, sequence_len, (nfilts*2+1)))
            encoding1 = encoder.predict(seq)[-1]
            encoding1 = np.vstack([encoding1]).reshape((1, 1, encodingN))
            repeater1 = np.repeat(encoding1, sequence_len, axis=1)
            out_seq = np.reshape(seq[:, :, 0], (len(seq), sequence_len, 1))
            lms_test = np.reshape(np.repeat(lms[i], sequence_len), (len(seq), -1))
            out_seq = np.dstack((out_seq, lms_test))

            decoding_input2 = np.concatenate((repeater1, out_seq), axis=-1)

            decoding2 = decoder.predict(decoding_input2)[0]

            plt.plot(seq[0, :, 0], seq[0, :, 1], 'o',color='green', alpha=1.0, linewidth=1)
            plt.plot(seq[0, :, 0], decoding2[:, 0], 'green', alpha=0.2, linewidth=10)
            plt.plot(seq[0, :, 0], seq[0, :, 2], 'o',color='red', alpha=1.0, linewidth=1)
            plt.plot(seq[0, :, 0], decoding2[:, 1], 'red', alpha=0.2, linewidth=10)
            plt.title(ids[i])
            #plt.plot(seq[0, :, 0], seq[0, :, 3], 'orange', alpha=1.0, linewidth=1)
            #plt.plot(seq[0, :, 0], decoding2[:, 2], 'orange', alpha=0.2, linewidth=10)
            #plt.plot(seq[0, :, 0], seq[0, :, 4], 'purple', alpha=1.0, linewidth=1)
            #plt.plot(seq[0, :, 0], decoding2[:, 3], 'purple', alpha=0.2, linewidth=10)
            plt.show()

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

        self.abs_mags = -2.5 * np.log10(self.fluxes) + self.zpt - 5. * np.log10(dist*1e6/10.0) + k_correction - reddening
        self.abs_mags_err = np.abs((2.5/np.log(10))*(self.flux_errs/self.fluxes))

        if replace_nondetections:
            abs_lim_mag = self.lim_mag - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction
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

def str2bool(v):
    """
    Helper function to turn strings to bool

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def feat_from_raenn(data_file, model_base=None,
                    prep_file=None, plot=False):
    """
    Calculate RAENN features

    Parameters
    ----------
    data_file : str
        Name of data file with light curves
    model_base : str
        Name of RAENN model file
    prep_file : str
        Name of file which encodes the feature prep

    Returns
    -------
    encodings : numpy.ndarray
        Array of object IDs (strings)

    TODO
    ----
    - prep file seems unnecessary
    """
    sequence, outseq, ids, maxlen, nfilts = prep_input(data_file, load=True, 
                                                       prep_file=prep_file)
    model_file = model_base + 'model.json'
    model_weight_file = model_base+'model.h5'
    with open(model_file, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weight_file)

    encodingN = model.layers[4].output_shape[1]
    input_1 = Input((None, nfilts*2+1))
    encoded_input = Input(shape=(None,(encodingN+2)))
    decoder_layer2 = model.layers[-3]
    decoder_layer3 = model.layers[-2]
    decoder_layer4 = model.layers[-1]

    merged = model.layers[-4]
    repeater = model.layers[-5]
    encoded2 = model.layers[2]
    encoded1 = model.layers[1]
    encoded = model.layers[3]
    encoded_sig = model.layers[4]
    decoder = Model(encoded_input,decoder_layer4(decoder_layer3(decoder_layer2(encoded_input))))
    encoder = Model(input_1, encoded(encoded2(encoded1(input_1))))
    encoder_sig = Model(input_1, encoded_sig(encoded2(encoded1(input_1))))

    if plot:
        decoder = get_decoder(model, encodingN)
        lms = outseq[:, 0, 1]
        sequence_len = maxlen
        get_decodings(decoder, encoder, sequence, lms, encodingN, sequence_len)

    encodings = np.zeros((len(ids), encodingN))
    for i in np.arange(len(ids)):
        if i%10000==0:
            print(i)
        inseq = np.reshape(sequence[i, :, :], (1, maxlen, nfilts*2+1))
        my_encoding = encoder.predict(inseq)
        encodings[i, :] = my_encoding
        encoder.reset_states()
    return encodings


def feat_peaks(input_lcs):
    """
    Extract peak magnitudes from GP LCs

    Parameters
    ----------
    input_lcs : list
        List of LC objects

    Returns
    -------
    peaks : list
        Peaks from each LC filter

    Examples
    --------
    """
    peaks = []
    for input_lc in input_lcs:
        peaks.append(np.nanmin(input_lc.dense_lc[:, :, 0], axis=0))
    return peaks


def feat_rise_and_decline(input_lcs, n_mag, nfilts=4):

    t_falls_all = []
    t_rises_all = []

    for i, input_lc in enumerate(input_lcs):
        gp = input_lc.gp
        gp_mags = input_lc.gp_mags
        t_falls = []
        t_rises = []
        for j in np.arange(nfilts):
            new_times = np.linspace(-100, 100, 500)
            x_stacked = np.asarray([new_times, [j] * 500]).T
            pred, var = gp.predict(gp_mags, x_stacked)

            max_ind = np.nanargmin(pred)
            max_mag = pred[max_ind]
            max_t = new_times[max_ind]
            trise = np.where((new_times < max_t) & (pred > (max_mag + n_mag)))
            tfall = np.where((new_times > max_t) & (pred > (max_mag + n_mag)))
            if len(trise[0]) == 0:
                trise = np.max(new_times) - max_t
            else:
                trise = max_t - new_times[trise][-1]
            if len(tfall[0]) == 0:
                tfall = max_t - np.min(new_times)
            else:
                tfall = new_times[tfall][0] - max_t

            t_falls.append(tfall)
            t_rises.append(trise)
        t_falls_all.append(t_falls)
        t_rises_all.append(t_rises)
    return t_rises_all, t_falls_all


def feat_slope(input_lcs, t_min_lim=10, t_max_lim=30, nfilts=4):
    slopes_all = []
    for i, input_lc in enumerate(input_lcs):
        gp = input_lc.gp
        gp_mags = input_lc.gp_mags
        slopes = []
        for j in np.arange(nfilts):
            new_times = np.linspace(-100, 100, 500)
            x_stacked = np.asarray([new_times, [j] * 500]).T
            pred, var = gp.predict(gp_mags, x_stacked)
            max_ind = np.nanargmin(pred)
            max_t = new_times[max_ind]
            new_times = new_times - max_t
            lc_grad = np.gradient(pred, new_times)
            gindmean = np.where((new_times > t_min_lim) & (new_times < t_max_lim))
            slopes.append(np.nanmedian(lc_grad[gindmean]))
        slopes_all.append(slopes)
    return slopes_all


def feat_int(input_lcs, nfilts=4):
    ints_all = []
    for i, input_lc in enumerate(input_lcs):
        gp = input_lc.gp
        gp_mags = input_lc.gp_mags
        ints = []
        for j in np.arange(nfilts):
            new_times = np.linspace(-100, 100, 500)
            x_stacked = np.asarray([new_times, [j] * 500]).T
            pred, var = gp.predict(gp_mags, x_stacked)
            ints.append(np.trapz(pred))

        ints_all.append(ints)
    return ints_all


def save_features(features, ids, feat_names, outputfile, outdir):
    # make output dir
    outputfile = outputfile+'.npz'
    outputfile = outdir + outputfile
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    np.savez(outputfile, features=features, ids=ids, feat_names=feat_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lcfile', type=str, help='Light curve file')
    parser.add_argument('--resample', type=str2bool, help='resampling')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data (single file)')
    parser.add_argument('--plot', type=str2bool, default=False, help='Plot LCs, for testing')
    parser.add_argument('--model-base', type=str, dest='model_base', default='./products/models/', help='...')
    parser.add_argument('--get-feat-raenn', type=str2bool, dest='get_feat_raenn', default=True, help='...')
    parser.add_argument('--get-feat-peaks', type=str2bool, dest='get_feat_peaks', default=False, help='...')
    parser.add_argument('--get-feat-rise-decline-1', type=str2bool,
                        dest='get_feat_rise_decline1', default=False,
                        help='...')
    parser.add_argument('--get-feat-rise-decline-2', type=str2bool,
                        dest='get_feat_rise_decline2', default=False,
                        help='...')
    parser.add_argument('--get-feat-rise-decline-3', type=str2bool,
                        dest='get_feat_rise_decline3', default=False,
                        help='...')
    parser.add_argument('--get-feat-slope', type=str2bool, dest='get_feat_slope', default=False, help='...')
    parser.add_argument('--get-feat-int', type=str2bool, dest='get_feat_int', default=False, help='...')
    parser.add_argument('--prep-file', type=str, dest='prep_file', default='./products/prep.npz', help='...')
    parser.add_argument('--outfile', type=str, dest='outfile', default='feat', help='...')

    args = parser.parse_args()
    features = []

    input_lcs = np.load(args.lcfile, allow_pickle=True)['lcs']
    ids = []
    feat_names = []
    for input_lc in input_lcs:
        if type(input_lc) == float:
            continue
        ids.append(input_lc.name)
    if args.get_feat_raenn:
        feat = feat_from_raenn(args.lcfile, model_base=args.model_base,
                               prep_file=args.prep_file, plot=args.plot,) 
                               #resample=args.resample)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('raenn'+str(i))
        print('RAENN feat done')

    if args.get_feat_peaks:
        feat = feat_peaks(input_lcs)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('peak'+str(i))
        print('peak feat done')

    if args.get_feat_rise_decline1:
        feat1, feat2 = feat_rise_and_decline(input_lcs, 1)
        if features != []:
            features = np.hstack((features, feat1))
            features = np.hstack((features, feat2))
        else:
            features = np.hstack((feat1, feat2))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('rise1'+str(i))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('decline1'+str(i))
        print('dur1 feat done')

    if args.get_feat_rise_decline2:
        feat1, feat2 = feat_rise_and_decline(input_lcs, 2)
        if features != []:
            features = np.hstack((features, feat1))
            features = np.hstack((features, feat2))
        else:
            features = np.hstack((feat1, feat2))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('rise2'+str(i))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('decline2'+str(i))
        print('dur2 feat done')

    if args.get_feat_rise_decline3:
        feat1, feat2 = feat_rise_and_decline(input_lcs, 3)
        if features != []:
            features = np.hstack((features, feat1))
            features = np.hstack((features, feat2))
        else:
            features = np.hstack((feat1, feat2))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('rise3'+str(i))
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('decline3'+str(i))
        print('dur3 feat done')

    if args.get_feat_slope:
        feat = feat_slope(input_lcs)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('slope'+str(i))
        print('slope feat done')

    if args.get_feat_int:
        feat = feat_int(input_lcs)
        if features != []:
            features = np.hstack((features, feat))
        else:
            features = feat
        for i in np.arange(np.shape(feat)[-1]):
            feat_names.append('int'+str(i))
        print('int feat done')

    if args.outdir[-1] != '/':
        args.outdir += '/'
#     save_features(features, ids, feat_names, args.outfile+'_'+date, outdir=args.outdir)
    save_features(features, ids, feat_names, args.outfile, outdir=args.outdir)


main()

