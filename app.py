import os
import pickle
import librosa
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from scipy.stats import skew, kurtosis
from pychorus import find_and_output_chorus
from flask import Flask, request, json, render_template



# Create flask app
app = Flask(__name__)

# Load pkl model
model = pickle.load(open('Ada_model', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    
    song_link = list(request.form.values())[0]
    
    # get features from songs
    data = []
    d, cols = extract_features(song_link)
    data.append(d)
    dataset = pd.DataFrame(data, columns=cols)




    dataset1 = pd.read_csv("Deployment/Data/feature_extract_04.csv")
    from pca import pca
    X = dataset1.drop('Artist', axis=1).drop('Title',axis=1).drop('Label', axis=1).drop('song_path', axis=1).drop('extracted_chorus_path', axis=1)
    # Initialize to reduce the data up to the nubmer of componentes that explains 95% of the variance.
    mod = pca(normalize=True)

    # Fit transform
    out = mod.fit_transform(X)

    df = pd.DataFrame(out['topfeat'])



    # select features which we used in ml model
    columns = list(df['feature'][df['type']=='best'])
    X = dataset[columns]

    # making prediction
    prediction = model.predict(X)
    output = 'Unpopular' if prediction[0] == 0 else 'Popular'
    return render_template('index.html', prediction_text = f'The song is {output}')


def statistics(list, feature, columns_name, data):
    i = 0
    for ele in list:
        _skew = skew(ele)
        columns_name.append(f'{feature}_kew_{i}')
        min = np.min(ele)
        columns_name.append(f'{feature}_min_{i}')
        max = np.max(ele)
        columns_name.append(f'{feature}_max_{i}')
        std = np.std(ele)
        columns_name.append(f'{feature}_std_{i}')
        mean = np.mean(ele)
        columns_name.append(f'{feature}_mean_{i}')
        median = np.median(ele)
        columns_name.append(f'{feature}_median_{i}')
        _kurtosis = kurtosis(ele)
        columns_name.append(f'{feature}_kurtosis_{i}')

        i += 1
        data.append(_skew) 
        data.append(min)
        data.append(max) 
        data.append(std) 
        data.append(mean) 
        data.append(median) 
        data.append(_kurtosis)
    
    return data

def extract_features(audio_path):

    data = []
    columns_name = []
    x , sr = librosa.load(audio_path)

    chroma_stft = librosa.feature.chroma_stft(x, sr)
    statistics(chroma_stft, 'chroma_stft', columns_name, data)

    chroma_cqt = librosa.feature.chroma_cqt(x, sr)
    statistics(chroma_cqt, 'chroma_cqt', columns_name, data)

    chroma_cens = librosa.feature.chroma_cens(x, sr)
    statistics(chroma_cens, 'chroma_cens', columns_name, data)

    mfcc = librosa.feature.mfcc(x, sr)
    statistics(mfcc, 'mfcc', columns_name, data)
    
    rms = librosa.feature.rms(x, sr)
    statistics(rms, 'rms', columns_name, data)

    spectral_centroid = librosa.feature.spectral_centroid(x , sr)
    statistics(spectral_centroid, 'spectral_centroid', columns_name, data)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(x , sr)
    statistics(spectral_bandwidth, 'spectral_bandwidth', columns_name, data)

    spectral_contrast = librosa.feature.spectral_contrast(x , sr)
    statistics(spectral_contrast, 'spectral_contrast', columns_name, data)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(x , sr)
    statistics(spectral_rolloff, 'spectral_rolloff', columns_name, data)

    tonnetz = librosa.feature.tonnetz(x , sr)
    statistics(tonnetz, 'tonnetz', columns_name, data)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(x , sr)
    statistics(zero_crossing_rate, 'zero_crossing_rate', columns_name, data)

    return data, columns_name
 

if __name__ == "__main__":
    app.run(debug=True)