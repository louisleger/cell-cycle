import os
import time
import bisect
import tifffile
import numpy as np
import pandas as pd
import multiprocessing as mp

"""

Welcome to the cell track extraction script. This script extracts the raw cell image values and fluorescent channel values into a directory for each well.

The input parameters are:
- EXPERIMENT and PATH to get the right directory with the raw cell videos and tracking data
- SAVE_PATH for the output directory
- CELL_SIZE the size of the square cut-out image to get of the cell.
- MIN_TRACK_LENGTH the minimum length of a cell track to be extracted 
- IMG_CHANNELS the channels of images to keep 
- N_CPU number of CPU the script can use to parallelize the cell extraction

The outputs created are 2 pickled dataframes per well: channels.pkl and images.pkl
Each row of the output dataframe corresponds to a cell at a specific time point in the video and it's information such as the position and area and it's integrated channels values or the cell images.

"""

EXPERIMENT  = 1
PATH = {
    1:"/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/260124-RPE-timelapse/Stitched/",
    2:"/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/21062024_RPE_before_after_inhibitors/Aligned/"
}
SAVE_PATH = f"/media/maxine/c8f4bcb2-c1fe-4676-877d-8e476418f5e5/0-RPE-cell-timelapse/extraction_z_score/"
CELL_SIZE = 64
MIN_TRACK_LENGTH = 72 # cell tracked for atleast 6h 
IMG_CHANNELS = [0, 3, 4]   #0: PC, 1: FG, 2: FR, 3: BF, 4: H2B
N_CPU = 16

"""
start of script; nothing left to modify
"""

# Paths to get raw video files and trackmate outputs res_track.txt
RAW_TIFF_PATH = PATH[EXPERIMENT]
TRACKMATE_PATH = PATH[EXPERIMENT] + "Tracked/CTC_results/"

# Main extraction function, for a number of masks, we get key information from raw miscroscope feed such as the cell position, radius, the cell image & integrated values
def get_cells_collection(mask_filenames, well_id, cells_of_interest):
        # Load raw video file with memmap
        ctc_results_directory = TRACKMATE_PATH+ f"{well_id}/01_RES/"
        I = tifffile.memmap(RAW_TIFF_PATH+f"{well_id}.tif")
        
        # Iterate through mask files produced by trackmate and get corresponding raw data from video file
        cell_stats_collection = {}
        for file in mask_filenames: 
            if (file.split('.')[1] == 'tif'):
                print(file)
                frame_idx = int(file.split('.')[0].split('k')[1])
                mask_img = tifffile.memmap(ctc_results_directory + file)

                # Consider only filtered cells and cells in the mask
                unique_cell_ids = np.unique(mask_img)
                frame_cells = set(cells_of_interest) & set(unique_cell_ids)
                background = np.mean(I[frame_idx], axis=(1, 2)); background_noise = np.std(I[frame_idx], axis=(1, 2))
                for cell_id in frame_cells:
                    if (cell_id not in cell_stats_collection.keys()):
                        cell_stats_collection[cell_id] =  {'AREA': [], 'POSITION': [], 'RADIUS': [], 'CHANNEL_SUM': [], 'CHANNEL_MEAN': [], 'CELL_IMG': [], 'FRAME': []}
                    
                    rows, cols = np.where(mask_img == cell_id)
                    area = len(rows)
                    position = np.around((np.mean(rows), np.mean(cols)), decimals=1)
                    distances = np.sqrt((rows - position[0])**2 + (cols - position[1])**2)
                    radius = np.around(np.max(distances), decimals=1)
                    
                    #Add statistics to cell collection
                    cell_stats_collection[cell_id]['AREA'].append(area)
                    cell_stats_collection[cell_id]['POSITION'].append(position)
                    cell_stats_collection[cell_id]['RADIUS'].append(radius)
                    cell_stats_collection[cell_id]['FRAME'].append(frame_idx)
                    
                    #Add integrated channel values(raw_frame - background)[:, np.newaxis, np.newaxis]/background_noise[:, np.newaxis, np.newaxis]

                    nucleus_values = np.round((I[frame_idx, :, rows, cols] - background[np.newaxis, :])/background_noise[np.newaxis, :], 2).astype(np.float16)
                    cell_stats_collection[cell_id]['CHANNEL_SUM'].append(nucleus_values.sum(axis = 0))
                    cell_stats_collection[cell_id]['CHANNEL_MEAN'].append(np.around(nucleus_values.mean(axis = 0), 2))
                    
                    #Add raw cutout of cell image
                    raw_img = I[frame_idx, IMG_CHANNELS, int(position[0]-CELL_SIZE/2):int(position[0]+CELL_SIZE/2), int(position[1]-CELL_SIZE/2):int(position[1]+CELL_SIZE/2)]
                    cell_stats_collection[cell_id]['CELL_IMG'].append(raw_img)

        print("Done Calculating Collection")
        return cell_stats_collection

def process_partition(args):
    cell_stats_collection = get_cells_collection(args[0], args[1], args[2])
    return cell_stats_collection

# Parallelized execution of get_cells_collection and dataframe saving function for each well
def optimized_expand_ctc_results(well_id, save_path):

    # Load txt file from trackmate output that gives you an edge list of CELL_IDs.
    # CELL_IDs are integers that correspond to the integers of the cell in segmentation masks over time. 
    ctc_results_directory = TRACKMATE_PATH + f"{well_id}/01_RES/"
    res_track = pd.read_csv(ctc_results_directory + "res_track.txt", sep = " ", header= None, names = ["CELL_ID", "T_START", "T_END", "PARENT"])
    print("Results from Tracking, Edge List Dimensions:", res_track.shape) #(number of cells/tracks, 4)

    # Filter out tracks that are very short with MIN_TRACK_LENGTH constant above
    res_track['DELTA_T'] = res_track.apply(lambda row: row['T_END']-row['T_START'], axis=1)
    filtered_tracks = res_track[res_track['DELTA_T'] > MIN_TRACK_LENGTH].copy()
    cells_of_interest = np.array(filtered_tracks['CELL_ID'].values.tolist())
    print(f"Tracks remaining after time filtering = {round(len(filtered_tracks)/len(res_track), 2)} %")

    # There is a mask for each frame of the video and since cells divide the last mask has exponentially more cells that the first mask
    # So each CPU shouldn't get an equal amount of masks as the first few masks are easy to process
    # To calculate how many masks each CPU should get, we compute the cost of each mask with the number of cells per mask

    for root, dirs, files in sorted(os.walk(ctc_results_directory)): mask_filenames = sorted(files)
    n_cells_per_frame = pd.DataFrame({1: np.zeros(len(mask_filenames), dtype=int)}).merge(
                        pd.DataFrame(filtered_tracks.apply(lambda cell: np.arange(cell['T_START'], cell['T_END']+1), axis=1).explode().reset_index().groupby(0).size()),
                        left_index=True, right_index=True).apply(lambda ncells: ncells[0] + ncells[1], axis=1).values.tolist()
    cost_per_file = n_cells_per_frame
    total_cost = np.sum(cost_per_file)
    cumsum_cost = np.cumsum(cost_per_file)
    ideal_cost_per_partition = total_cost//N_CPU
    print("Total Cost:", total_cost)
    print("Ideal cost per split:", ideal_cost_per_partition)
    partitions = []
    start_idx = 0
    for i in range(1, N_CPU):
        target_cost = ideal_cost_per_partition * i
        end_idx = bisect.bisect_left(cumsum_cost, target_cost, lo=start_idx)
        partitions.append(mask_filenames[start_idx:end_idx])
        print("Partition indexes", start_idx, end_idx)
        start_idx = end_idx
    partitions.append(mask_filenames[start_idx:])

    # Each partition is a list of a masks for 1 CPU to extract cell data from
    # Create a pool of processes
    pool = mp.Pool(processes=N_CPU)

    # Process each partition in parallel
    lst_cell_collections = pool.map(process_partition,  [(partition, well_id, cells_of_interest) for partition in partitions])

    # Close the pool to release resources
    pool.close()
    pool.join()

    # Merge collections of cell data extracted from each CPU
    print("Merging") 
    def merge_stats_collections(lst_cell_collections):
        merged_stats = {}
        for cell_collections in lst_cell_collections:
            for cell_id, stats in cell_collections.items():
                if cell_id not in merged_stats:
                    merged_stats[cell_id] = stats
                else:
                    for key, value in stats.items():
                        if key in merged_stats[cell_id]:
                            merged_stats[cell_id][key].extend(value)
                        else:
                            merged_stats[cell_id][key] = value
        return merged_stats
    cell_stats_collection = merge_stats_collections(lst_cell_collections)
    print("Done Merging") 

    # Add the collections data to the dataframe
    random_cell = list(cell_stats_collection.keys())[0]
    stat_names = cell_stats_collection[random_cell].keys()
    for key in stat_names:  
        filtered_tracks[key] = filtered_tracks['CELL_ID'].apply(lambda cid: cell_stats_collection[cid][key] if cid in cell_stats_collection.keys() else np.nan)
    
    # Explode dataframe for readability so each row is 1 cell at 1 dt, inverse operation is .groupby
    result_df = pd.DataFrame()
    cells = filtered_tracks['CELL_ID'].values.tolist()
    for c in cells:
        exploded_df = filtered_tracks[filtered_tracks['CELL_ID'] == c].explode(column = ['FRAME','AREA','POSITION',
                        'RADIUS','CHANNEL_SUM', 'CHANNEL_MEAN', 'CELL_IMG']
                        ).reset_index(drop=True)
        result_df = pd.concat([result_df, exploded_df], axis=0).reset_index(drop=True)
    
    # Separate extracted data into 2 output dataframes and save/pickle them
    column_types = {"information": ["T_START", "T_END", "DELTA_T", "AREA", "POSITION", "RADIUS", "PARENT"],
                    "channel": ['CHANNEL_SUM', 'CHANNEL_MEAN'],
                    "images": ['CELL_IMG'],}
    
    channel_df = result_df[['CELL_ID', 'FRAME'] + column_types["channel"] + column_types['information']]
    channel_df.to_pickle(save_path + f"channels.pkl")

    image_df = result_df[['CELL_ID', 'FRAME'] + column_types["images"]]
    image_df.to_pickle(save_path + f"images.pkl")
    return result_df

#Execute script
if __name__ == '__main__':

    # Get the wells of an experiment
    wells = sorted([well.split('.')[0] for well in os.listdir(PATH[EXPERIMENT]) if well.endswith('.tif')])
    for well_id in wells: 
        start_time = time.time()
        
        final_save_path = SAVE_PATH + f"{well_id}/"
        os.makedirs(final_save_path, exist_ok=True)
        #Run parallelized extraction
        results = optimized_expand_ctc_results(well_id, final_save_path)

        end_time = time.time()
        print(f"Done well {well_id} in {(end_time - start_time)/60} minutes, file: {SAVE_PATH}")
    print(f"All Done!")
    