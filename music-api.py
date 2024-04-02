from fastapi import FastAPI, File, UploadFile
import io
import os
import glob
import librosa
import numpy as np
from python_speech_features import mfcc
import pickle
from annoy import AnnoyIndex
from tqdm import tqdm

app = FastAPI()

data_dir = './music'  # Thay đổi đường dẫn đến thư mục chứa dữ liệu âm thanh của bạn

def extract_features(y, sr=16000, nfilt=10, winstep=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winstep)
        return feat
    except Exception as e:
        print("Extraction feature error:", e)
        return None

def crop_feature(feat, i=0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat

features = []
songs = []

# Load or create and save features and songs
if os.path.exists('features.pk') and os.path.exists('songs.pk'):
    features = pickle.load(open('features.pk', 'rb'))
    songs = pickle.load(open('songs.pk', 'rb'))
else:
    for song_file in glob.glob(os.path.join(data_dir, '*.mp3')):
        try:
            y, sr = librosa.load(song_file, sr=16000)
            feat = extract_features(y)
            for i in range(0, feat.shape[0] - 10, 5):
                features.append(crop_feature(feat, i, nb_step=10))
                songs.append(song_file)
        except Exception as e:
            print("Error processing file:", song_file, e)
    
    # Save features and songs to files
    pickle.dump(features, open('features.pk', 'wb'))
    pickle.dump(songs, open('songs.pk', 'wb'))

# Build Annoy index
f = 100
t = AnnoyIndex(f, 'euclidean')  # Assuming euclidean distance
for i in range(len(features)):
    v = features[i]
    t.add_item(i, v)
t.build(100)  # 100 trees
t.save('music.ann')

# Load Annoy index
u = AnnoyIndex(f, 'euclidean')
u.load('music.ann')

# Function to find similar songs for a given audio snippet
def find_similar_songs(audio_path, winstep=0.02, nb_step=10):
    y, sr = librosa.load(audio_path, sr=16000)
    feat = extract_features(y, sr=sr, winstep=winstep)
    if feat is None:
        print("Unable to extract features from audio:", audio_path)
        return {"error": "Unable to extract features from audio"}
    results = []
    for i in range(0, feat.shape[0], nb_step):
        crop_feat = crop_feature(feat, i, nb_step=nb_step)
        result = u.get_nns_by_vector(crop_feat, n=5)
        result_songs = [songs[k] for k in result]
        results.extend(result_songs)
    if  (len(results) == 0):
        return {"error": "No similar songs found"}
    else:
        return {"similar_songs": results[0]}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    with open("./uploaded_file.mp3", "wb") as f:
        f.write(contents)
    
    similar_songs = find_similar_songs("./uploaded_file.mp3")
    return {"similar_songs": similar_songs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

