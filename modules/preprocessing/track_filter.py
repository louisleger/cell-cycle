import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

"""

Welcome to the track filtering script. This script filters out bad tracks and outputs a directory with images and labels of train and test tracks saved as .npy

The input parameters are: 
- EXTRACTION_DIRECTORY where the pickled dataframes from the extraction script are
train_wells
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
    'mtor': ['0706', '0707', '0708', '0709'],
    'drugged': [f'{i:02d}' +  f'{j:02d}' for i in range(3, 6) for j in range(4, 12)] + [f'{i:02d}' +  f'{2:02d}' for i in range(3, 7)] + ['0706', '0707', '0708', '0709'],
}

# Full tracks to keep
full_tracks = set(pd.read_csv('/home/maxine/Documents/louis/perfect_repo/full_tracks.csv', dtype='str')['CELL_ID'].tolist())

# Parameters where we specify which wells to get in the training or test portion
folder_name= "/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/track_datasets/control/"
train_wells= wells_dictionnary['healthy']
test_wells= wells_dictionnary['healthy_test']
cell_size= 64
minmax = True

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

    # Scaling integrated fluorescent signals to be used as labels
    # 0: PC, 1: F1/G, 2: F2/R, 3: BF, 4: H2B
    channels['fucci1'] = channels.apply(lambda cell: cell['CHANNEL_SUM'][1]/cell['AREA'], axis=1) 
    channels['fucci2'] = channels.apply(lambda cell: cell['CHANNEL_SUM'][2]/cell['AREA'], axis=1) 
    epsilon =  np.array([np.percentile(channels['fucci1'].values, q=1), np.percentile(channels['fucci2'].values, q=1)])
    softplus = lambda x, beta=1: np.where(beta*x < 20, (1/beta)*np.log(1+np.exp(beta*x)), x)
    channels['fucci1'] = channels['fucci1'].apply(lambda f1: np.log2( softplus( (f1 - epsilon[0])/epsilon[0])))
    channels['fucci2'] = channels['fucci2'].apply(lambda f2: np.log2( softplus( (f2 - epsilon[1])/epsilon[1])))

    df = pd.merge(channels[['CELL_ID', 'FRAME', 'fucci1', 'fucci2', 'POSITION', 'PARENT']],
                   images[['CELL_IMG']], left_index=True, right_index=True)

    # Verifying Image Sizes
    df['correct_size'] = df['CELL_IMG'].apply(lambda im: True if im.shape[1] == cell_size and
                                              im.shape[2] == cell_size else False)
    border_cells = np.unique(df[df['correct_size'] == False]['CELL_ID'].values)
    df = df[~df['CELL_ID'].isin(border_cells)].reset_index(drop=True)
    print("Cells w/ wrong image sizes (border cells):", len(border_cells), 'or',  round(len(border_cells)/number_of_cells, 2), '%')
    
    # Standardizing image values per image, so the pixel values are normally distributed
    print("Standardizing Images")
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
                                                    'CELL_IMG': group['CELL_IMG'].values.tolist(),
                                                    'POSITION': group['POSITION'].values.tolist(),
                                                    'PARENT': group['PARENT'].values.tolist()[0], }))).reset_index()
    
    #Filtering out partial tracks with list of full tracks
    print('Filtering out partial tracks')
    cell_life['full'] = cell_life['CELL_ID'].apply(lambda cid: cid in full_tracks)
    cell_life = cell_life[cell_life['full'] == True].reset_index(drop=True).copy()
    
    if (minmax):
        print('Min-Max')
        cell_life['fucci1'] = cell_life['fucci1'].apply(lambda f1: mm.fit_transform(np.array(f1).reshape(-1, 1)))
        cell_life['fucci2'] = cell_life['fucci2'].apply(lambda f2: mm.fit_transform(np.array(f2).reshape(-1, 1)))

    # Getting tau the normalized time, tau = time/track_length = t/T
    cell_life['tau'] = cell_life['frames'].apply(lambda frame_list: np.linspace(0, 1, len(frame_list)))
    print("Total Trainable Cells:", len(cell_life), 'or', round(len(cell_life)/number_of_cells, 2), '%')

    # Save interesting information for later in the form of well meta data
    metadata = {
        'CELL_ID': cell_life['CELL_ID'].values.tolist(),
        'FRAMES': cell_life['frames'].values.tolist(), 
        'POSITION': [np.array(p).tolist() for p in cell_life['POSITION'].values.tolist()], 
    }
    with open(f'{save_path}{well}_meta.json', 'w') as f:
        json.dump(metadata, f)

    # Save track track images and labels in respective folders
    print('Saving...')
    os.makedirs(save_path + 'images/', exist_ok=True)
    cell_life.apply(lambda track: np.save(f"{save_path}images/{track['CELL_ID']}.npy", arr=np.array(track['CELL_IMG'])), axis=1)
    os.makedirs(save_path + 'labels/', exist_ok=True)
    cell_life.apply(lambda track: np.save(f"{save_path}labels/{track['CELL_ID']}.npy", arr=np.stack([track['fucci1'], track['fucci2']])), axis=1)
    return well

#Execute script
if __name__ == '__main__':
    os.makedirs(folder_name, exist_ok=True)
    
    os.makedirs(folder_name + 'train/', exist_ok=True)
    for well in train_wells:
        print(well)
        filter_exploded_dataframes(well, save_path = folder_name + 'train/')

    os.makedirs(folder_name + 'test/', exist_ok=True)
    for well in test_wells:
        print(well)
        filter_exploded_dataframes(well, save_path = folder_name + 'test/')

    print(f"All Done!")