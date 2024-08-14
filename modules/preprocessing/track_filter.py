import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""

Welcome to the track filtering script. This script filters out bad tracks and outputs a directory with images and labels of train and test tracks saved as .npy

The input parameters are: 
- EXTRACTION_DIRECTORY where the pickled dataframes from the extraction script are
- CONFIG dictionnary where we specify certain filtering and dataset preparation parameters:
    - folder_name: output folder name
    - train wells: list of well IDs for training
    - test wells: list of well IDs for testing
    - label_measure: make ground truth labels of fluorescent channels with a certain label measure
    - label_transform: in which sort coordinates are we observing these fluorescent label channels: 'log' or 'linear'
    - image_normalization: how should we normalize the image values, 'local': per image standaridization
    - parents: boolean to consider only tracks that go through cytokinesis
    - cell_size: cell_size constant from track_extraction.py

The output directory will be a collection of .npy files containing image tensors of shape (T, C, H, W) where T is track length, C is number of channels, H, W are image dimensions

"""

# Extraction directory is location of well directories with outputs from track extraction script
EXTRACTION_DIRECTORY = "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/extraction/"

# Wells dictionnary is a dictionnary to group certain wells with same starting conditions like control cells, or drugged cells
wells_dictionnary = { 
    'healthy': ['0403', '0503', '0603', '0607', '0608', '0609', '0610', '0611'],    
    'healthy_test': ['0303'],
    'mTOR_inhibitor': ['0706', '0707', '0708', '0709'],
    'drugged': [f'{i:02d}' +  f'{j:02d}' for i in range(3, 6) for j in range(4, 12)] + [f'{i:02d}' +  f'{2:02d}' for i in range(3, 7)] + ['0706', '0707', '0708'],
    'drugged_test': ['0709'],
    'complete': ['0403', '0503', '0603', '0607', '0608', '0609', '0610', '0611'] + [f'{i:02d}' +  f'{j:02d}' for i in range(3, 6) for j in range(4, 11)] +  ['0706', '0707', '0709'] + [f'{i:02d}' +  f'{2:02d}' for i in range(3, 6)],
    'complete_test': ['0708'] + ['0303'] + ['0602'] + [f'{i:02d}' +  f'{11:02d}' for i in range(3, 6)]
}

# Configuration dictionnary where we can specify which train wells, if we are transforming the labels to log, what type of track filtering
CONFIG = {
    "folder_name": "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/track_datasets/complete_full_test/",
    "train_wells": wells_dictionnary['complete'],
    "test_wells": wells_dictionnary['complete_test'],
    "label_measure": "CHANNEL_SUM",
    "label_transform": "log",
    "image_normalization": "local", 
    "parents": True,
    "cell_size": 64,
    "filtering_function": "full" # "neg": negative correlation or "full": full fucci red profiles 
}

# Function to apply a logarithm to a numpy array with a normalized output (Min-Max)
def log_transform(x, epsilon=1e-1):
    mm = MinMaxScaler()
    x = np.where(x < 0, 0, x)
    tx = np.log(x + epsilon)
    y =  mm.fit_transform(tx.reshape(-1, 1)).flatten()
    return y

# Function to winsorize outlier values of a numpy array
def winsor_outlier_removal(x, tolerance=2.5):
    Q1, Q3 = np.percentile(x, q=25), np.percentile(x, q=75)
    IQR = Q3-Q1
    upper, lower = Q3 + tolerance*IQR,  Q1 - tolerance*IQR

    x[x > upper] = upper
    x[x < lower] = lower
    return x

# Function to filter the cells of a well and save them in a folder for deep learning
def filter_exploded_dataframes(well, save_path):
    
    # Check if data exists
    if not os.path.exists(EXTRACTION_DIRECTORY + well + "/channels.pkl"): return well
    
    # Loading Data
    channels = pd.read_pickle(EXTRACTION_DIRECTORY + well + "/channels.pkl")
    images = pd.read_pickle(EXTRACTION_DIRECTORY + well + f"/images.pkl")
    
    # Changing the name of each track to be associated with it's well
    channels['CELL_ID'] = channels['CELL_ID'].apply(lambda cid: well + '.' + str(cid))
    images['CELL_ID'] = images['CELL_ID'].apply(lambda cid: well + '.' +str(cid))
    
    number_of_cells = len(np.unique(channels['CELL_ID'].values))
    print('Loaded', number_of_cells, 'cells')

    # Retreiving integrated fluorescent signals to be used as labels
    # 0: PC, 1: F1/G, 2: F2/R, 3: BF, 4: H2B
    channels['fucci1'] = channels[CONFIG['label_measure']].apply(lambda cha: cha[1]) 
    channels['fucci2'] = channels[CONFIG['label_measure']].apply(lambda cha: cha[2]) 
    channels['h2b']    = channels[CONFIG['label_measure']].apply(lambda cha: cha[4])

    df = pd.merge(channels[['CELL_ID', 'FRAME', 'fucci1', 'fucci2', 'h2b', 'POSITION', 'PARENT']],
                   images[['CELL_IMG']], left_index=True, right_index=True)

    # Verifying Image Sizes    
    df['correct_size'] = df['CELL_IMG'].apply(lambda im: True if im.shape[1] == CONFIG['cell_size'] and
                                              im.shape[2] == CONFIG['cell_size'] else False)
    border_cells = np.unique(df[df['correct_size'] == False]['CELL_ID'].values)
    df = df[~df['CELL_ID'].isin(border_cells)].reset_index(drop=True)
    print("Cells w/ wrong image sizes (border cells):", len(border_cells), 'or',  round(len(border_cells)/number_of_cells, 2), '%')
    
    # Standardizing image values per image, so the pixel values are normally distributed
    print("Standardizing Images and Normalizing Reporters Trackwise")
    def standardize_per_channel(cimg):
        cimg_st = np.zeros_like(cimg, dtype=np.float32)
        channels = cimg.shape[0]
        for cdx in range(channels):
            cimg_st[cdx] = (cimg[cdx] - np.mean(cimg[cdx]))/(np.std(cimg[cdx]) + 1e-6)
        return cimg_st
    df['CELL_IMG'] = df['CELL_IMG'].apply(lambda cimg: standardize_per_channel(cimg))

    # Syntax to have a grouped dataframe, where each row is all the time points of a cells
    cell_life = pd.DataFrame(df.groupby('CELL_ID').apply(lambda group: pd.Series({'frames': group['FRAME'].values.tolist(), 
                                                    'fucci1': group['fucci1'].values.tolist(),
                                                    'fucci2': group['fucci2'].values.tolist(),
                                                    'h2b': group['h2b'].values.tolist(),
                                                    'CELL_IMG': group['CELL_IMG'].values.tolist(),
                                                    'POSITION': group['POSITION'].values.tolist(),
                                                    'PARENT': group['PARENT'].values.tolist()[0], }))).reset_index()
    
    # Min Max scaling FUCCI values to 0 - 1 range
    mm = MinMaxScaler()
    for reporter in ['fucci1', 'fucci2', 'h2b']:
        cell_life[reporter] = cell_life[reporter].apply(lambda r: mm.fit_transform(np.array(r).reshape(-1, 1)).flatten())

    # Tranforms the fucci from linear to log scale
    if (CONFIG['label_transform'] == "log"):
        cell_life['fucci1'] = cell_life['fucci1'].apply(lambda r: log_transform(r, epsilon=1e-2)) 
        cell_life['fucci2'] = cell_life['fucci2'].apply(lambda r: log_transform(r, epsilon=1e-1))

    #Filtering out with negative FUCCI correlation and Double Cytokinesis
    print('Filtering out negative correlation and double cytokinesis')
    cell_life['correlation'] = cell_life.apply(lambda track: np.corrcoef(np.array(track['fucci1']), np.array(track['fucci2']))[0, 1], axis=1)
    cell_life['full_fucci_red'] = cell_life['fucci2'].apply(lambda fr: np.abs(fr[-1] - fr[0]) < 0.5)
    parents = set(cell_life.apply(lambda track: track['CELL_ID'].split('.')[0] + '.' + str(track['PARENT']), axis=1).values.tolist())
    cell_life['is_parent'] = cell_life['CELL_ID'].apply(lambda cid: int(cid in parents)); cell_life['is_daughter'] = cell_life['PARENT'].apply(lambda p: int(p > 0))
    cell_life['cytokinesis'] = cell_life.apply(lambda track: track['is_parent'] << 1 | track['is_daughter'], axis=1)
    if (CONFIG['filtering_function'] == 'neg'):
        cell_life = cell_life[(cell_life['cytokinesis'] == 3) & (cell_life['correlation'] < 0)].reset_index(drop=True).copy()
    else: 
        cell_life = cell_life[(cell_life['cytokinesis'] == 3) & (cell_life['full_fucci_red'] == True)].reset_index(drop=True).copy()


    # Getting tau the normalized time, tau = time/track_length = t/T
    cell_life['tau'] = cell_life['frames'].apply(lambda frame_list: np.linspace(0, 1, len(frame_list)))
    print("Total Trainable Cells:", len(cell_life), 'or', round(len(cell_life)/number_of_cells, 2), '%')

    # Save interesting information for later in the form of well meta data
    metadata = { 
        'CELL_ID': cell_life['CELL_ID'].values.tolist(),
        'FRAMES': cell_life['frames'].values.tolist(), 
        'POSITION': [np.array(p).tolist() for p in cell_life['POSITION'].values.tolist()], 
        'h2b':  cell_life['frames'].values.tolist(), 
    }
    with open(f'{save_path}{well}_meta.json', 'w') as f:
        json.dump(metadata, f)

    # Save track track images and labels in respective folders
    os.makedirs(save_path + 'images/', exist_ok=True)
    cell_life.apply(lambda track: np.save(f"{save_path}images/{track['CELL_ID']}.npy", arr=np.array(track['CELL_IMG'])), axis=1)
    os.makedirs(save_path + 'labels/', exist_ok=True)
    cell_life.apply(lambda track: np.save(f"{save_path}labels/{track['CELL_ID']}.npy", arr=np.stack([track['fucci1'], track['fucci2']])), axis=1)
    return well

#Execute script
if __name__ == '__main__':
    os.makedirs(CONFIG['folder_name'], exist_ok=True)
    
    os.makedirs(CONFIG['folder_name'] + 'train/', exist_ok=True)
    for well in CONFIG['train_wells']:
        print(well)
        filter_exploded_dataframes(well, save_path = CONFIG['folder_name'] + 'train/')
        print('Saving...')
    
    os.makedirs(CONFIG['folder_name'] + 'test/', exist_ok=True)
    for well in CONFIG['test_wells']:
        print(well)
        filter_exploded_dataframes(well, save_path = CONFIG['folder_name'] + 'test/')
        print('Saving...')

    print(f"All Done!")