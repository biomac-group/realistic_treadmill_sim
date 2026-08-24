import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal

def get_data(p, n_points=100):
    grf = p.grf_data
    speed = p.beltspeed_data
    fs_speed = 1/np.diff(speed.index).mean()
    speed_filtered = filter_data(speed, fs = fs_speed, fc=30)
    cycles = average_cycles(grf.GRF_y_1, grf.GRF_y_2, grf.GRF_x_1, grf.GRF_x_2, speed_filtered.left, speed_filtered.right, 
            n_points=n_points)
    dur, std = get_cycle_duration(grf.GRF_y_1)
    return cycles, dur


def filter_data(data, fs, fc):
        b, a = signal.butter(2, fc, fs = fs)
        column_names = data.columns
        index = data.index
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        filtered_data = pd.DataFrame(filtered_data, columns=column_names)
        filtered_data.index = index
        return filtered_data


def get_cycle_duration(fy1):
    contact = np.where(fy1 > 10, 1, 0)
    hs = []
    to = []

    for i in range(1,len(contact)):
        if contact[i - 1] == 1 and contact[i] == 0:
            to.append(fy1.index[i])
        if contact[i] == 1 and contact[i - 1] == 0:
            hs.append(fy1.index[i])

    durations = []
    for i in range(len(hs)-1):
        
        cycle_start = hs[i]
        cycle_end = hs[i+1]
        
        if (cycle_end - cycle_start) < 0.6 or (cycle_end - cycle_start) > 1.5:
            continue

        
        duration = cycle_end - cycle_start
      
        durations.append(duration)
    mean_dur = np.mean(durations)
    std_dur = np.std(durations)
    return mean_dur, std_dur
    
    
def average_cycles(fy1, fy2, fx1, fx2, vel_left, vel_right, n_points = 100):
    contact = np.where(fy1 > 10, 1, 0)
    hs = []
    to = []

    for i in range(1,len(contact)):
        if contact[i - 1] == 1 and contact[i] == 0:
            to.append(fy1.index[i])
        if contact[i] == 1 and contact[i - 1] == 0:
            hs.append(fy1.index[i])


    all_cycles_vel_left = []
    all_cycles_fy1 = []
    all_cycles_fx1 = []
    all_cycles_vel_right = []
    all_cycles_fy2 = []
    all_cycles_fx2 = []
    durations = []
    for i in range(len(hs)-1):
        
        cycle_start = hs[i]
      
        cycle_end = hs[i+1]
        if (cycle_end - cycle_start) < 0.6 or (cycle_end - cycle_start) > 1.5:
            continue
  
        
        fy1_data = fy1[cycle_start:cycle_end]
        fx1_data = fx1[cycle_start:cycle_end]
        vel_data1 = vel_left[cycle_start:cycle_end]

        f = interp1d(vel_data1.index, vel_data1)
        x_new = np.linspace(vel_data1.index[0], vel_data1.index[-1], num=n_points, endpoint=True)
        interpolated_vel1 = f(x_new)

        f = interp1d(fy1_data.index, fy1_data)
        x_new = np.linspace(fy1_data.index[0], fy1_data.index[-1], num=n_points, endpoint=True)
        interpolated_fy1 = f(x_new)

        f = interp1d(fx1_data.index, fx1_data)
        x_new = np.linspace(fx1_data.index[0], fx1_data.index[-1], num=n_points, endpoint=True)
        interpolated_fx1 = f(x_new)

        all_cycles_vel_left.append(interpolated_vel1)
        all_cycles_fy1.append(interpolated_fy1)
        all_cycles_fx1.append(interpolated_fx1)

      
        fy2_data = fy2[cycle_start:cycle_end]
        fx2_data = fx2[cycle_start:cycle_end]
        vel_data2 = vel_right[cycle_start:cycle_end]


        f = interp1d(vel_data2.index, vel_data2)
        x_new = np.linspace(vel_data2.index[0], vel_data2.index[-1], num=n_points, endpoint=True)
        interpolated_vel2 = f(x_new)

        f = interp1d(fy2_data.index, fy2_data)
        x_new = np.linspace(fy2_data.index[0], fy2_data.index[-1], num=n_points, endpoint=True)
        interpolated_fy2 = f(x_new)

        f = interp1d(fx2_data.index, fx2_data)
        x_new = np.linspace(fx2_data.index[0], fx2_data.index[-1], num=n_points, endpoint=True)
        interpolated_fx2 = f(x_new)


        all_cycles_vel_right.append(interpolated_vel2)
        all_cycles_fy2.append(interpolated_fy2)
        all_cycles_fx2.append(interpolated_fx2)
        

 
    mean1 = np.mean(np.array(all_cycles_vel_left), axis=0)
    mean2 = np.mean(np.array(all_cycles_vel_right), axis=0)
    mean3 = np.mean(np.array(all_cycles_fy1), axis=0)
    mean4 = np.mean(np.array(all_cycles_fy2), axis=0)
    mean5 = np.mean(np.array(all_cycles_fx1), axis=0)
    mean6 = np.mean(np.array(all_cycles_fx2), axis=0)
    

    
    df =  pd.DataFrame([mean1, mean2, mean3, mean4, mean5, mean6]).T
    df.columns = ['speedL', 'speedR', 'FyL', 'FyR', 'FxL', 'FxR']
    return df





