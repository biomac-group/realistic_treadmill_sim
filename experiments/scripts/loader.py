import os
import glob
from tpcp import Dataset
from typing import List, Optional, Union
from itertools import   product
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal


def get_all_participant_IDs(main_folder):
    participant_folders = []

    # Traverse through all subdirectories and collect participant folder names
    for root, dirs, _ in os.walk(main_folder):
        for folder in dirs:
            participant_folders.append(folder)

    return participant_folders


from scipy import signal
from functools import lru_cache

#@lru_cache(maxsize=1)
def _extract_file_metadata(base_path, config_path):
    final_files = {}
    config = pd.read_excel(config_path, header = 0, index_col = 0)

    for p in sorted(base_path.rglob("*.mot")):
        *_, pat_id, _, measurement = p.parts
        measurement = measurement.split(".")[0]

        if 'static' in measurement:
            speed = 0
            task = 'static'
        
        else:
            m_number = int(measurement[7:8])
        
            speed = config.loc[pat_id, m_number]
            task = 'walking'
        
        
      
        final_files[(pat_id, speed, measurement, task)] = p

    return final_files


def _extract_file_metadata_csv(base_path, config_path):
    final_files = {}
    config = pd.read_excel(config_path, header = 0, index_col = 0)

    for p in sorted(base_path.rglob("*.csv")):
        *_, pat_id, _, measurement = p.parts
        measurement = measurement.split(".")[0]
       # print(measurement)

        m_number = int(measurement[-1])
        
        speed = config.loc[pat_id, m_number]
        task='other'

        
        
        
      
        final_files[(pat_id, speed, measurement, task)] = p

    return final_files


class DatasetGrail(Dataset):
    def __init__(self, data_path,
                 config_path,
                 use_lru_cache: bool = True,
                 groupby_cols: Optional[Union[List[str], str]] = None,
                 subset_index: Optional[pd.DataFrame] = None,
                 ):
        self.data_path = data_path
        self.use_lru_cache = use_lru_cache
        self.config_path = config_path

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)


    @property
    def _file_metadata(self):
        return _extract_file_metadata(Path(self.data_path), self.config_path)

  


    def _get_file_from_metadata(self, metdata_tuple):
        return self._file_metadata[metdata_tuple]


    def create_index(self):
        final_files = pd.DataFrame(list(self._file_metadata.keys()), columns=["id", "speed", "measurement", "task"])
        return final_files


    #----------functions related to LOADING the data-----------------------


    @property
    def marker_data(self):
        #load marker data
        self.assert_is_single(None, "recording")
        id = self.index['id'][0]
        task = self.index['task'][0]
        marker_names = pd.read_csv(self.data_path + '/' + id +'/' + task  + '.trc', index_col=None, delimiter='\t', skiprows=3, nrows=1, header=None).dropna(axis=1).values.flatten().tolist()
        marker_names_detailed = [item + suffix for item in marker_names[2:] for suffix in ['_X', '_Y', '_Z']]
        marker = pd.read_csv(self.data_path + '/' + id +'/' + task  + '.trc', index_col=None, delimiter='\t', skiprows=5, header=None)
        marker.columns = ['Frame', 'Time', *marker_names_detailed]

        marker.interpolate(method='linear', inplace=True, axis=0)
        time = marker.Time

        marker = self.filter_data(marker, 100)
        marker['Time'] = time
        marker['Frame'] = np.arange(1, len(marker)+1)


        #cut marker data to the task of interest
        return marker

    @property
    def grf_data_downsampled(self):
        #load grf data
        self.assert_is_single(None, "recording")
        id = self.index['id'][0]
        task = self.index['measurement'][0]
        grf = self.load_grf_file(self.data_path + '/' +  id +'/' + task  + '.mot')
        dict = {'ground_force1_vx': 'GRF_x_1', 'ground_force1_vy': 'GRF_y_1', 'ground_force1_vz': 'GRF_z_1',
                'ground_force1_px': 'COP_x_1', 'ground_force1_py': 'COP_y_1', 'ground_force1_pz': 'COP_z_1',
                'ground_torque1_x': 'M_x_1', 'ground_torque1_y': 'M_y_1', 'ground_torque1_z': 'M_z_1',
                'ground_force2_vx': 'GRF_x_2', 'ground_force2_vy': 'GRF_y_2', 'ground_force2_vz': 'GRF_z_2',
                'ground_force2_px': 'COP_x_2', 'ground_force2_py': 'COP_y_2', 'ground_force2_pz': 'COP_z_2',
                'ground_torque2_x': 'M_x_2', 'ground_torque2_y': 'M_y_2', 'ground_torque2_z': 'M_z_2'}
        grf.rename(columns=dict, inplace=True)
    



        return grf.iloc[0::10, :].reset_index()
    
    @property
    def grf_data(self):
        #load grf data
        self.assert_is_single(None, "recording")
        id = self.index['id'][0]
        task = self.index['measurement'][0]
        pattern = os.path.join(self.data_path, '**', id, '**', f'{task}.mot')
       
        matches = glob.glob(pattern, recursive=True)
        grf = self.load_grf_file(matches[0])
        dict = {'ground_force1_vx': 'GRF_z_1', 'ground_force1_vy': 'GRF_y_1', 'ground_force1_vz': 'GRF_x_1',
                'ground_force1_px': 'COP_z_1', 'ground_force1_py': 'COP_y_1', 'ground_force1_pz': 'COP_x_1',
                'ground_torque1_x': 'M_z_1', 'ground_torque1_y': 'M_y_1', 'ground_torque1_z': 'M_x_1',
                'ground_force2_vx': 'GRF_z_2', 'ground_force2_vy': 'GRF_y_2', 'ground_force2_vz': 'GRF_x_2',
                'ground_force2_px': 'COP_z_2', 'ground_force2_py': 'COP_y_2', 'ground_force2_pz': 'COP_x_2',
                'ground_torque2_x': 'M_z_2', 'ground_torque2_y': 'M_y_2', 'ground_torque2_z': 'M_x_2'}
        grf.rename(columns=dict, inplace=True)
        grf.set_index('Time', inplace=True, drop=True)
        if grf.index[-1] > 7:
            return grf.loc[10:]
        else:
            return grf

    @property
    def beltspeed_data(self):
        self.assert_is_single(None, "recording")
        id = self.index['id'][0]
        task = self.index['measurement'][0]
        task = "W" + task[1:]
        pattern = os.path.join(self.data_path, '**', id, '**', f'{task}.txt')
       
        matches = glob.glob(pattern, recursive=True)
       
        tr_speed = pd.read_csv(matches[0], index_col = 0, delimiter = '\t')
        tr_speed = tr_speed.iloc[:,[0,2]]
        tr_speed.columns = ['left', 'right']



        return tr_speed.loc[10:]
    
        

    def load_grf_file(self, filename):
        grf = pd.read_csv(filename, index_col = None, delimiter='\t', skiprows=6)
        columns = grf.columns
        grf = grf.reset_index(inplace=False).iloc[:,:-1]
        grf.columns = columns




        time = grf.time
        grf = self.filter_data(grf, 1000)
        grf['Time'] = time
        grf = grf.drop('time', axis=1)

        return grf

    @property
    def data(self):
        return pd.concat([self.marker_data, self.grf_data], axis=1, join='inner')

    @property
    def mass(self):
        config = pd.read_excel(self.config_path, header = 0, index_col = 0)
        id = self.index['id'][0]
        return config.loc[id, 'mass']

    @property
    def height(self):
        config = pd.read_excel(self.config_path, header = 0, index_col = 0)
        id = self.index['id'][0]
        return config.loc[id, 'height']
        

    
    def filter_data(self, data, fs):
        b, a = signal.butter(2, 10, fs = fs)
        column_names = data.columns
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        filtered_data = pd.DataFrame(filtered_data, columns=column_names)
        return filtered_data


    @property
    def grf_data_raw(self):
        #load grf data
        self.assert_is_single(None, "recording")
        id = self.index['id'][0]
        task = self.index['measurement'][0]
        pattern = os.path.join(self.data_path, '**', id, '**', f'{task}.mot')
       
        matches = glob.glob(pattern, recursive=True)
        grf = pd.read_csv(matches[0], index_col = None, delimiter='\t', skiprows=6)
        columns = grf.columns
        grf = grf.reset_index(inplace=False).iloc[:,:-1]
        grf.columns = columns




        time = grf.time

        grf['Time'] = time
        grf = grf.drop('time', axis=1)

      
        dict = {'ground_force1_vx': 'GRF_z_1', 'ground_force1_vy': 'GRF_y_1', 'ground_force1_vz': 'GRF_x_1',
                'ground_force1_px': 'COP_z_1', 'ground_force1_py': 'COP_y_1', 'ground_force1_pz': 'COP_x_1',
                'ground_torque1_x': 'M_z_1', 'ground_torque1_y': 'M_y_1', 'ground_torque1_z': 'M_x_1',
                'ground_force2_vx': 'GRF_z_2', 'ground_force2_vy': 'GRF_y_2', 'ground_force2_vz': 'GRF_x_2',
                'ground_force2_px': 'COP_z_2', 'ground_force2_py': 'COP_y_2', 'ground_force2_pz': 'COP_x_2',
                'ground_torque2_x': 'M_z_2', 'ground_torque2_y': 'M_y_2', 'ground_torque2_z': 'M_x_2'}
        grf.rename(columns=dict, inplace=True)
        grf.set_index('Time', inplace=True, drop=True)
        return grf.loc[10:]


