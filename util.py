# Description: This file contains all the utility functions used in the project.
# %%
# Initialization

from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
import scipy.signal as signal

import scipy.fft as fft

from torch.utils.data import Dataset, DataLoader
import re
import copy


# %%
def read_fits_file(file, bin=False, noisemode="GaussianNoise", snr=None):
    '''
    Read a FITS file and return a pandas DataFrame
    Parameters:
    -----------
    file: str
        Path to the FITS file
    bin: bool or float
        If True, bin the data to a specified bin size (in days). If False, do not bin the data. If a float is provided, bin the data to that bin size (in seconds).
    noisemode: str
        The type of noise to add to the data. Can be "GaussianNoise", "PoissonNoise", "FullScaleNoise" or None. If None, no noise will be added.
    snr: float
        The signal-to-noise ratio to use when adding noise. If None, no noise will be added. If a float is provided, the noise will be added based on the specified SNR.
    Returns:
    --------
    df: pandas.DataFrame
        The light curve data as a pandas DataFrame
    '''
    fits_file = fits.open(file)
    df = Table(fits_file[1].data).to_pandas()
    df = df.interpolate(method='linear', axis=0)

    attr = df.keys()
    attr = attr[2:]
    if noisemode == "GaussianNoise": 

        if snr not in [None, False]:
            if snr == True:
                snr = np.random.choice([10.,5.,3.,1.,0.1])
            
            #print(snr)
            # attr = df.keys()
            # attr = attr[2:]

            for k in attr:
                '''
                Add noise based on Sacchi's lecture
                Noise contains Gaussian noise with \sigma = \sqrt{Es/(snr*En)}
                '''
                noise = np.random.randn(len(df))
                Es = np.sum(df[k]**2)
                En = np.sum(noise**2)
                alpha = np.sqrt(Es/(snr*En))
                df[k] = np.maximum(df[k] + alpha * noise, 0)

    elif noisemode == "PoissonNoise":
        for k in attr:
            '''
            Noise contains only Poisson noise with \lambda = x(t).
            '''
            df[k] = np.random.poisson(lam=df[k])

    elif noisemode == "FullScaleNoise":

        if snr in [None, False]:
            raise Exception("SNR is not specified. Please specify SNR or set it to False.")
        else:                
            for k in attr:
                '''
                    First of all, Poisson Noise
                '''
                df[k] = np.random.poisson(lam=df[k])

                '''
                Add noise based on Sacchi's lecture
                Noise contains Gaussian noise with \sigma = \sqrt{Es/(snr*En)}
                '''
                noise = np.random.randn(len(df))
                Es = np.sum(df[k]**2)
                En = np.sum(noise**2)
                alpha = np.sqrt(Es/(snr*En))
                df[k] = np.maximum(df[k] + alpha * noise, 0)

    elif noisemode == None:
        pass

    else:
        raise Exception("Mode not found. Please choose from GaussianNoise or PoissonNoise.")

    if bin != False:
        if type(bin) in [float, int]:     
            df = binning(df,bin_size=float(bin))
        else:
            raise ValueError("Bin Size must be specified! and it must be float or int.")

    return df

def read_lightcurves(LightCurves, mode="Full", mock_up=False, snr = None, bin=False, noisemode="PoissonNoise"):
    '''
    Read a list of FITS files and return a numpy array of light curves
    Parameters:
    -----------
    LightCurves: list
        List of paths to the FITS files
    mode: str
        The mode of the light curves to read. Can be "Soft", "Hard" or "Full". If "Soft", only the soft X-ray light curves will be read. If "Hard", only the hard X-ray light curves will be read. If "Full", both soft and hard X-ray light curves will be read.
    mock_up: bool
        If True, generate mock-up light curves with a given SNR and binning. If False, the original light curves will be used.
    snr: float
        The signal-to-noise ratio to use when adding noise. If None, no noise will be added. If a float is provided, the noise will be added based on the specified SNR
    bin: bool or float
        If True, bin the data to a specified bin size (in days). If False, do not bin the data. If a float is provided, bin the data to that bin size (in seconds).
    noisemode: str
        The type of noise to add to the data. Can be "GaussianNoise", "PoissonNoise", "FullScaleNoise" or None. If None, no noise will be added.
    Returns:
        --------
        X: numpy.array
            The concatenated light curves.
    '''
    Xs = []
    Xh = []
    for x in LightCurves:
        df = read_fits_file(x,bin=bin,snr=snr,noisemode=noisemode)
        if mode == "Soft" or mode == "Full":
            Xs.append(df["lc_s"].values)
        if mode == "Hard" or mode == "Full":
            Xh.append(df["lc_h"].values)

    Xs = np.array(Xs)
    Xh = np.array(Xh)

    # mock up real data
    if mock_up:
        N_nulls = 72
        N = len(df)
        sigma = 10
        locations = np.random.randint(0, N-sigma, N_nulls)
        for i in locations:
            size = int(np.random.normal(60, sigma))
            Xs[i:i+size+1] = 0
            Xh[i:i+size+1] = 0

    if mode == "Full":
        X = np.concatenate([Xs, Xh], axis=0)
    elif mode == "Soft":
        X = Xs
    elif mode == "Hard":
        X = Xh
    else:
        raise Exception("Mode not found. Please choose from Full, Soft or Hard.")

    X = torch.tensor(X, dtype=torch.float32)
    X = nn.functional.normalize(X)
    return X

def read_response(Responses, mode="Full"):
    '''
    Read a list of FITS files and return a numpy array of response functions
    Parameters:
    -----------
    Responses: list
        List of paths to the FITS files for the response functions.
    mode: str
        The mode of the response functions to read. Can be "Soft", "Hard" or "Full". If "Soft", only the soft X-ray response functions will be read. If "Hard", only the hard X-ray response functions will be read. If "Full", both soft and hard X-ray response functions will be read.
    Returns:
    --------
    Y: torch.Tensor
        The concatenated response functions.
    '''
    Y = []
    for y in Responses:
        df = read_fits_file(y)
        Y.append(df["response"].values)
    if mode == "Full":    
        Y = np.concatenate([Y,Y],axis=0)
    elif mode == "Soft" or mode == "Hard":
        Y = np.array(Y)
    else:
        raise Exception("Mode not found. Please choose from Full, Soft or Hard.")
    
    Y = torch.tensor(Y, dtype=torch.float32)
    Y = nn.functional.normalize(Y)

    return Y

def binning(df, bin_size):
    """
    Binning the time series data into bins of size bin_size (in seconds). The binning is done by taking the mean of the values in each bin.
    Parameters:
    -----------
    df: pandas.DataFrame
        The light curve data as a pandas DataFrame
    bin_size: float
        The size of the bins in seconds.
    Returns:
    --------
    df: pandas.DataFrame
        The binned light curve data as a pandas DataFrame
    Note:
    i.e. 0.5 TU means 2 observations per TU
         1 TU means 1 observation per TU
         2 TU means 1 observation for every 2 TU
    """
    try:
        df['bin'] = df['time'] // bin_size
    except:
        df['bin'] = df['TIME'] // bin_size
    df = df.groupby('bin').mean()
    df['time'] = df.index * bin_size
    return df

def load_model(model_name,model,optimizer):
    '''
    Load a trained model from a checkpoint file.
    Parameters:
    -----------
    model_name: str
        The name of the model to load. The checkpoint file should be located in the "Trained_Model" directory and should have the name "{model_name}.pt".
    model: torch.nn.Module
        The model architecture to load the weights into.
    optimizer: torch.optim.Optimizer
        The optimizer to load the state into.
    Returns:
    --------
    model: torch.nn.Module
        The model with the loaded weights.
    optimizer: torch.optim.Optimizer
        The optimizer with the loaded state.
    epoch: int
        The epoch at which the model was saved.
    loss_history: list
        The loss history of the model during training.
    '''
    checkpoint = torch.load(f"Trained_Model/{model_name}.pt",weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss']
    return model, optimizer, epoch, loss_history

def extract_parameters(LC):
    '''
    Extract the scale height and the response function radius from the light curve filename.
    Parameters:
    -----------
    LC: str
        The filename of the light curve. The filename should be in the format "lc_h{scale_height}_Rs{response_function_radius}.fits".
    Returns:
    --------
    h: float
        The scale height extracted from the filename.
    Rs: float
        The response function radius extracted from the filename.
    '''
    filename = LC.split("/")[-1]
    h, Rs = re.findall(r"[-+]?\d*\.\d+|\d+", filename)[0:3:2]
    #h, Rs = LC.split(".")[0].split("_")[1:4:2]
    h = float(h)
    Rs = float(Rs)
    return h, Rs

def p_percent_rule(X, p=0.8):
    '''
    Apply the p-percent rule to a numpy array. The p-percent rule sets all values in the array that are less than p times the maximum value to zero.
    Parameters:
    -----------
    X: numpy.array
        The input array to apply the p-percent rule to.
    p: float
        The percentage of the maximum value to use as the threshold. Values less than p times the maximum value will be set to zero. Default is 0.8 (80%).
    Returns:
    --------
    X: numpy.array
        The array after applying the p-percent rule.
    '''
    max_val = np.max(X)
    X[np.where(X < p * max_val)] = 0
    return X

def extract_cc_amplitude(ground_truth, predicted):
    '''
    Extract the cross-correlation amplitude between the ground truth and predicted response functions.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth response function.
    predicted: torch.Tensor or numpy.array
        The predicted response function.
    Returns:
    --------
    cc_amplitude: float
        The cross-correlation amplitude between the ground truth and predicted response functions.
    '''
    if ground_truth.dtype == torch.float:    
        y = ground_truth.numpy().copy()

    else:
        y = np.array(ground_truth.copy())
    y = y/np.linalg.norm(y)
    #y -= np.mean(y)
    
    if predicted.dtype == torch.float:
        y_star = predicted.numpy().copy()
    else:
        y_star = np.array(predicted.copy())
    y_star = y_star/np.linalg.norm(y_star)
    #y_star -= np.mean(y_star)

    cc = signal.correlate(y_star, y, mode='full')
    lags = signal.correlation_lags(len(y_star), len(y))

    #return cc[len(cc)//2]
    return cc[np.where(lags==0.)][0]


def extract_cc_amplitude_fourier(ground_truth, predicted):
    '''
    Extract the cross-correlation amplitude between the ground truth and predicted response functions using Fourier transform.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth response function.
    predicted: torch.Tensor or numpy.array
        The predicted response function.
    Returns:
    --------
    cc_amplitude: float
        The cross-correlation amplitude between the ground truth and predicted response functions.
    '''
    if ground_truth.dtype == torch.float:    
        y = ground_truth.numpy().copy()

    else:
        y = np.array(ground_truth.copy())

    y = fft.fft(y)
    y = np.abs(y)**2
    y = y/np.linalg.norm(y)
    
    if predicted.dtype == torch.float:
        y_star = predicted.numpy().copy()
    else:
        y_star = np.array(predicted.copy())
    y_star = fft.fft(y_star)
    y_star = np.abs(y_star)**2
    y_star = y_star/np.linalg.norm(y_star)
    
    cc = signal.correlate(y_star, y, mode='full')
    lags = signal.correlation_lags(len(y_star), len(y))

    return cc[np.where(lags==0.)][0]

def cosine_similarity(ground_truth, prediction):
    '''
    Compute the cosine similarity between two vectors.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth vector.
    prediction: torch.Tensor or numpy.array
        The predicted vector.
    Returns:
    --------
    cosine_similarity: float
        The cosine similarity between the ground truth and predicted vectors.
    '''
    if ground_truth.dtype == torch.float:
        actual = ground_truth.numpy().copy()
    else:
        actual = ground_truth
    if prediction.dtype == torch.float:
        predict = prediction.numpy().copy()
    else:
        predict = prediction
    # a_diff = actual - np.mean(actual)
    # p_diff = predic - np.mean(predic)
    numerator = np.dot(actual, predict)
    denominator = np.sqrt(np.sum(actual ** 2)) * np.sqrt(np.sum(predict ** 2))
    return numerator / denominator

def calc_mape(ground_truth, prediction, eplison = 1e-6):
    '''
    Calculate the Mean Absolute Percentage Error (MAPE) between the ground truth and predicted values.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth values.
    prediction: torch.Tensor or numpy.array
        The predicted values.
    eplison: float
        A small value added to the ground truth and predicted values to avoid division by zero. Default is 1e-6.
    Returns:
    --------
    mape: float
        The Mean Absolute Percentage Error (MAPE) between the ground truth and predicted values.
    '''
    actual = ground_truth.numpy().copy() + float(eplison)
    predic = prediction.numpy().copy() + float(eplison)

    return np.mean(np.abs((actual - predic) / actual))

def get_coherence(ground_truth, prediction):
    '''
    Compute the coherence between two signals.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth signal.
    prediction: torch.Tensor or numpy.array
        The predicted signal.
    Returns:
    --------
    coherence: float
        The coherence between the ground truth and predicted signals.
    '''
    if ground_truth.dtype == torch.float:
        A = copy.deepcopy(ground_truth.numpy())
    else:
        A = np.array(copy.deepcopy(ground_truth))
    A /= np.linalg.norm(A)

    if prediction.dtype == torch.float:
        B = copy.deepcopy(prediction.numpy())
    else:
        B = np.array(copy.deepcopy(prediction))
    B /= np.linalg.norm(B)

    freq, Cxy = signal.coherence(A,B)

    #return np.max(Cxy)
    #return Cxy[np.where(freq==0.)][0]
    return Cxy[0]
    #return np.nanmean(Cxy)

def get_centroid(timestamp,signal):
    '''
    Compute the centroid of a signal given its timestamp and signal values.
    Parameters:
    -----------
    timestamp: torch.Tensor or numpy.array
        The timestamps of the signal.
    signal: torch.Tensor or numpy.array
        The signal values.
    Returns:
    --------
    centroid: float
        The centroid of the signal.
    '''
    if type(timestamp) == torch.Tensor:
        t = timestamp.numpy().copy()
    else:
        t = timestamp.copy()
    if type(signal) == torch.Tensor:
        y = signal.numpy().copy()
    else:
        y = signal.copy()
    
    y = p_percent_rule(y)
        
    return np.sum(t*y)/np.sum(y)

def get_centroid_dist(ground_truth, predicted):
    '''
    Compute the distance between the centroids of the ground truth and predicted signals.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth signal.
    predicted: torch.Tensor or numpy.array
        The predicted signal.
    Returns:
    --------
    centroid_distance: float
        The distance between the centroids of the ground truth and predicted signals.
    '''
    t = np.arange(ground_truth.shape[0])
    gt_centroid = get_centroid(t, ground_truth)
    pred_centroid = get_centroid(t, predicted)
    return 1 - min(np.abs((gt_centroid - pred_centroid)/gt_centroid),1)

def write_report(indicator,key):
    '''
    Write a report of the indicator between the predicted and true response functions.
    Parameters:
    -----------
    indicator: numpy.array
        The indicator values between the predicted and true response functions.
    key: str
        The name of the indicator.
    '''
    assert type(indicator) == np.ndarray, "Indicator must be a numpy array."
    print(f"\
    {key} between predicted and true response functions is: {np.percentile(indicator, 50):.3f} (+ {(np.percentile(indicator,75) - np.percentile(indicator,50)):.3f} - {(np.percentile(indicator,50) - np.percentile(indicator,25)):.3f})")

def compute_similarity(ground_truth, predictions, indicators):
    '''
    Compute the similarity between the predicted and true response functions using the specified indicators.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth signal.
    predictions: torch.Tensor or numpy.array
        The predicted signal.
    indicators: dict
        A dictionary of indicator functions.
    Returns:
    --------
    similarity: dict
        A dictionary of similarity values for each indicator.
    '''
    similarity = {}
    
    for key in indicators.keys():
        similarity[key] = np.array([])
        
    for i in range(len(ground_truth)):
        for key in indicators.keys():
            similarity[key] = np.append(similarity[key],indicators[key](ground_truth[i], predictions[i]))
    
    for key in indicators.keys():
        write_report(similarity[key], key)

    return similarity


def distance_of_first_response(ground_truth, predictions):
    '''
    Compute the distance of the first response between the predicted and true response functions.
    Parameters:
    -----------
    ground_truth: torch.Tensor or numpy.array
        The ground truth signal.
    predictions: torch.Tensor or numpy.array
        The predicted signal.
    Returns:
    --------
    distance: float
        The distance of the first response between the predicted and true response functions.
    '''
    gt = ground_truth.numpy().copy()
    gt = np.round(gt,3)
    pred = predictions.numpy().copy()
    pred = np.round(pred,3)
    
    first_response_gt = np.where(gt > 0)[0][0]
    first_response_pred = np.where(pred > 0)[0][0]
    
    return np.abs(first_response_gt - first_response_pred)/first_response_gt