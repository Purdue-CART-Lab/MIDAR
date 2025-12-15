import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
import argparse
import sys

import torch

from MIDAR_helpers import _build_los_frame_for_cav, los_visible_ids_from_graph
sys.path.append('../../los_graphormer/models')
from los_graphormer import LoSGraphormer

# ===============================================================
# CAV Detections Generationd
# ===============================================================

def CP_data_generation(
        df: pd.DataFrame,
        device='cpu',
        gnn=None,
        occ_thresh: float = 0.22167,
        Range: float = 80.0,
        seed: int = 42,
        VIS_GRAPH: bool = False,
        bin_size: float = 10.0,
        detect_mode: str = "MIDAR",      # {"MIDAR","random_drop","perfect"}
        use_ray_hit: bool = True,
    ):
    """
    detect_mode:
      - "MIDAR":  use MIDAR (mimics CenterPoint detection)
      - "random_drop": use Random Drop (drop observations based on probabilities)
      - "perfect": all in-range vehicles are visible
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---------- stats buckets (optional) ----------
    bins         = np.arange(0, Range + bin_size, bin_size)
    vis_counts   = np.zeros(len(bins) - 1, dtype=int)
    total_counts = np.zeros(len(bins) - 1, dtype=int)

    # ---------- working copy ----------
    df = df.copy()
    df['Detected'] = 0

    # Ensure required columns exist (from trajectory dataset)
    required = ['Vehicle_ID','Global_Time','Global_X','Global_Y',
                'v_Width','v_Length','v_Height','yaw','CAV']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------- which CAVs to process ----------
    cav_ids = df.loc[df['CAV'] == 1, 'Vehicle_ID'].drop_duplicates().to_numpy()

    # ---------- mode selection ----------
    resolved_mode = detect_mode

    use_randomdrop = (resolved_mode == "random_drop")
    use_MIDAR       = (resolved_mode == "MIDAR")
    use_perfect   = (resolved_mode == "perfect")

    # ===================== main loop =====================
    for cav_id in cav_ids:
        cav_traj = df[df['Vehicle_ID'] == cav_id]

        for _, cav_row in cav_traj.iterrows():

            # Frame at this time
            t = cav_row['Global_Time']
            time_mask = (df['Global_Time'] == t)
            frame     = df[time_mask]

            # Relative distances wrt this CAV (absolute coords in df)
            dx   = frame['Global_X'].to_numpy(dtype=float) - float(cav_row['Global_X'])
            dy   = frame['Global_Y'].to_numpy(dtype=float) - float(cav_row['Global_Y'])
            dist = np.hypot(dx, dy)

            in_range   = dist <= float(Range)
            neigh_ids  = frame.loc[in_range, 'Vehicle_ID'].to_numpy()
            neigh_dist = dist[in_range]
            
            # If only ego is present, mark detected and continue
            if len(neigh_ids) == 1:
                df.loc[time_mask & (df['Vehicle_ID'] == int(cav_id)), 'Detected'] = 1
                continue

            # ---------------- perfect path ----------------
            if use_perfect:
                visible_ids = neigh_ids.tolist()

            # ---------------- MIDAR path ----------------
            elif use_MIDAR:
                # Absolute positions; builder recenters ego internally
                def _first_value(series):
                    return float(series.iloc[0])

                positions = {
                    int(v): (
                        _first_value(frame.loc[frame['Vehicle_ID'] == v, 'Global_X']),
                        _first_value(frame.loc[frame['Vehicle_ID'] == v, 'Global_Y'])
                    )
                    for v in neigh_ids
                }
                dims = {
                    int(v): tuple(frame.loc[frame['Vehicle_ID'] == v,
                                            ['v_Width','v_Length','v_Height']].iloc[0].astype(float))
                    for v in neigh_ids
                }
                def _yaw(v):
                    s = frame.loc[frame['Vehicle_ID'] == v, 'yaw']
                    return float(s.iloc[0]) if len(s) and not pd.isna(s.iloc[0]) else 0.0
                headings = {int(v): _yaw(v) for v in neigh_ids}

                data = _build_los_frame_for_cav(
                    int(cav_id),
                    [int(v) for v in neigh_ids if v != cav_id],
                    positions, dims, headings,
                    use_ray_hit = use_ray_hit
                ).to(device)

                try:
                    vis_nodes = los_visible_ids_from_graph(data, gnn, occ_thresh=occ_thresh)
                except Exception as e:
                    raise RuntimeError(
                        f"LoS-Graphormer forward failed at time {t} "
                        f"for CAV {int(cav_id)}: {e}"
                    )

                idx_to_vid = [int(cav_id)] + [int(v) for v in neigh_ids if v != cav_id]
                max_idx = max(vis_nodes)
                if max_idx >= len(idx_to_vid):
                    raise RuntimeError(
                        f"Index mapping mismatch: max vis node {max_idx}, "
                        f"but only {len(idx_to_vid)} IDs in idx_to_vid."
                    )
                
                visible_ids = [idx_to_vid[i] for i in sorted(vis_nodes)]

                # Diagnostics by range bin (optional)
                for vid, d in zip(neigh_ids, neigh_dist):
                    idx = np.digitize(d, bins, right=False) - 1
                    if 0 <= idx < len(total_counts):
                        total_counts[idx] += 1
                        if int(vid) in visible_ids:
                            vis_counts[idx] += 1

                if VIS_GRAPH:
                    ratios = np.divide(
                        vis_counts, total_counts,
                        out=np.full_like(vis_counts, np.nan, dtype=float),
                        where=total_counts > 0
                    )
                    print("\nVisibility ratio by detection range (GNN path):")
                    for i in range(len(bins) - 1):
                        print(f"{int(bins[i]):>3}-{int(bins[i+1]):<3} m : "
                              f"{ratios[i]:.3f}  ({vis_counts[i]}/{total_counts[i]})")

            # -------------- Random Drop Path --------------
            elif use_randomdrop:
                visible_ids = []
                for vid, d in zip(neigh_ids, neigh_dist):
                    # TPRs of CenterPoint on the custom CARLA dataset within different distance intervals
                    if   d < 10: p = 0.9902
                    elif d < 20: p = 0.9432
                    elif d < 30: p = 0.8864
                    elif d < 40: p = 0.8847
                    elif d < 50: p = 0.8150
                    elif d < 60: p = 0.7293
                    elif d < 70: p = 0.6354
                    elif d < 80: p = 0.5134
                    else:        p = 0.0
                    if rng.random() < p:
                        visible_ids.append(int(vid))

                # Ego must be visible to itself
                if int(cav_id) not in visible_ids:
                    visible_ids.append(int(cav_id))

            # Final write-back for this timestamp
            df.loc[time_mask & df['Vehicle_ID'].isin(visible_ids), 'Detected'] = 1

    return df

# ===============================================================
# Trajectory Reconstruction Preparation
# ===============================================================

def build_complete_trajectories_dict(df):
    """
    Build a dict where each key is the same as time_dict's key (e.g. 1, 2, 3...),
    and each value is a dictionary of {vehicle_id: subset_of_df_for_that_vehicle}
    for all vehicles whose entire trajectory is contained within the time range
    specified by time_dict[key] = [start_time, end_time].
    """
    
    def build_cav_time_dict(df):
        """
        Build a dictionary where:
          key: An integer (starting from 1)
          value: [enter_time_of_current_CAV, exit_time_of_next_CAV]
        
        Assumptions:
          - 'CAV' is 1 for those vehicles designated as CAV.
          - 'Vehicle_ID' identifies vehicles.
          - 'Global_Time' is the time column used for 'enter' and 'exit'.
        """

        # 1) Filter rows where CAV == 1
        df_cav = df[df['CAV'] == 1].copy()

        # 2) Identify each distinct CAV vehicle
        cav_ids = df_cav['Vehicle_ID'].unique()
        
        df_cav_whole = df[df['Vehicle_ID'].isin(cav_ids)].copy()

        # 3) For each CAV vehicle, find min and max time
        #    We'll store tuples of the form: (cav_id, enter_time, exit_time)
        cav_enter_exit = []
        for cid in cav_ids:
            df_cid = df_cav_whole[df_cav_whole['Vehicle_ID'] == cid]
            enter_time = df_cid['Global_Time'].min()
            exit_time  = df_cid['Global_Time'].max()
            cav_enter_exit.append((cid, enter_time, exit_time))

        # 4) Sort by the enter_time (the second element in each tuple)
        cav_enter_exit.sort(key=lambda x: x[1])  # sort by enter_time

        # 5) Build the dictionary
        #    my_dict[i] = [enter_time_of_i, exit_time_of_(i+1)]
        #    We skip the last CAV because it has no "next" one
        time_dict = {}
        for i in range(len(cav_enter_exit) - 1):
            # i-th CAV is cav_enter_exit[i]
            # next CAV is cav_enter_exit[i+1]
            curr_enter = cav_enter_exit[i][1]
            next_enter  = cav_enter_exit[i+1][1]
            time_dict[i + 1] = [curr_enter, next_enter]

        return time_dict
    
    time_dict = build_cav_time_dict(df)

    # 1) Compute the min and max Global_Time for each vehicle
    #    (i.e., the earliest and latest time that vehicle appears)
    grouped = df.groupby('Vehicle_ID')['Global_Time']
    min_times = grouped.min()

    # 2) Initialize the output dictionary
    complete_trajectories_dict = {}

    # 3) Iterate over time_dict
    #    time_dict[k] = [start_time, end_time]
    for k, (start_time, end_time) in time_dict.items():
        
        # 3.1) Find vehicles whose entire min->max time is fully in [start_time, end_time]
        #      min_time >= start_time AND max_time <= end_time
        valid_vehicles = min_times[
            (min_times >= start_time) & (min_times <= end_time)
        ].index
        
        # 3.2) Store in the output dictionary under key k
        complete_trajectories_dict[k] = df[df['Vehicle_ID'].isin(valid_vehicles)].copy()

    return complete_trajectories_dict

    
def CP_traj_visualization(data_all,lane_index,time_start,time_end, section_direction,index, legacy=True):

    #filter vehicles in the lane
    data_all_filtered=data_all[data_all['Lane_ID'] == lane_index]
    data_filtered = data_all_filtered[data_all_filtered['Detected'] == 1]

    data_all_filtered.reset_index(drop=True, inplace=True)
    data_filtered.reset_index(drop=True, inplace=True)
    
    first_GT = True
    first_cav = True
    first_detected = True
    
    plt.figure(dpi=200)
    
    #legacy
    if legacy:
        veh_id_all_list=list(set(data_all_filtered.Vehicle_ID))
        data_all_filtered.Global_Time = data_all_filtered['Global_Time'].round(2)
        
        
        for veh_id in veh_id_all_list:
            data_all_veh = data_all_filtered[data_all_filtered.Vehicle_ID==veh_id]
            
            time_veh=list(data_all_veh.Global_Time)
            
            time_points=[[time_veh[0]]]
            for i in range(0,len(time_veh)-1):
                if time_veh[i + 1] - time_veh[i] > 1:
                    time_points[-1].append(time_veh[i])
                    time_points.append([time_veh[i+1]])
            time_points[-1].append(time_veh[-1])
            
            for j in range(len(time_points)):
                segment = data_all_veh[(data_all_veh.Global_Time >= time_points[j][0]) & (data_all_veh.Global_Time <= time_points[j][1])]
                if first_GT:
                    plt.plot(segment['Global_Time'], segment['Local_Y'],color='chocolate', label='Undetected')
                    first_GT = False
                else:
                    plt.plot(segment['Global_Time'], segment['Local_Y'],color='chocolate')
    #cav and aug

    veh_id_list=list(set(data_filtered.Vehicle_ID))
    
    #print(veh_id_list)
    
    data_filtered.Global_Time = data_filtered['Global_Time'].round(2)
    
    for veh_id in veh_id_list:
        
        #print(veh_id)
        
        data_veh = data_filtered[data_filtered.Vehicle_ID == veh_id]
        if (data_veh['CAV'] == 1).any():
            color = 'red'
        else:
            color = 'blue'

        time_veh=list(data_veh.Global_Time)
        
        time_points=[[time_veh[0]]]
        for i in range(0,len(time_veh)-1):
            if time_veh[i + 1] - time_veh[i] > 1:
                time_points[-1].append(time_veh[i])
                time_points.append([time_veh[i+1]])
        time_points[-1].append(time_veh[-1])
        
        for j in range(len(time_points)):
            segment = data_veh[(data_veh.Global_Time >= time_points[j][0]) & (data_veh.Global_Time <= time_points[j][1])]
                
            if color == 'red':
                if first_cav:
                    #plt.scatter(segment['Global_Time'], segment['Local_Y'], color=color, s=20, label='CAV')
                    plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, linewidth=2, label='CAV')
                    first_cav = False
                else:
                    #plt.scatter(segment['Global_Time'], segment['Local_Y'], color=color, s=20)
                    plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, linewidth=2)
            elif color == 'blue':
                if first_detected:
                    #plt.scatter(segment['Global_Time'], segment['Local_Y'], color=color, s=20, label='Detected')
                    plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, linewidth=2, label='Detected')
                    first_detected = False
                else:
                    #plt.scatter(segment['Global_Time'], segment['Local_Y'], color=color, s=20)
                    plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, linewidth=2)

    plt.title('lane '+str(lane_index))
    plt.xlim(time_start,time_end)
    plt.ylim(0,420)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (meters)')
    plt.grid(True)
    plt.legend(loc='lower right')
    #plt.savefig('./results/'+section_direction+'/'+str(index)+'/ref_'+str(lane_index)+'.png')
    plt.show()

def visualize_traj(all_veh, index, section_direction, lane_list, legacy=True):
    #os.mkdir('./results/'+section_direction+'/'+str(index))
    
    for i in range(len(lane_list)):
        CP_traj_visualization(all_veh[index],lane_list[i], min(all_veh[index].Global_Time),
                              max(all_veh[index].Global_Time), section_direction, index, legacy=legacy)


#%% carla MIDAR detection
def run_MIDAR(data_path, ckpt_path, occ_thresh, use_ray_hit):

    traj_true = pd.read_csv(data_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    in_feats = 5 if use_ray_hit else 4
    
    if use_ray_hit:
        output_fname = "./CP_traj_data/carla_MIDAR.pkl"
    else:
        output_fname = "./CP_traj_data/carla_MIDAR_noRH.pkl"

    _gnn = LoSGraphormer(
        in_feats=in_feats,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=32,
        num_classes=2
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    _gnn.load_state_dict(ckpt, strict=False)
    _gnn.eval()

    # Call the generator
    traj_MIDAR = CP_data_generation(
        df=traj_true,
        device=device,
        gnn=_gnn,
        occ_thresh=occ_thresh,
        Range=80.0,
        seed=42,
        detect_mode='MIDAR',
        use_ray_hit = use_ray_hit,
    )

    traj_MIDAR_dict = build_complete_trajectories_dict(traj_MIDAR)
    with open(output_fname, "wb") as file:
        pickle.dump(traj_MIDAR_dict, file)
    #visualize_traj(traj_MIDAR_dict, 1, '1', [1, 2, 3, 4], legacy=True)

#%% carla true lidar detection
def run_true(data_path):
    traj_true = pd.read_csv(data_path)
    traj_true_dict = build_complete_trajectories_dict(traj_true)
    with open("./CP_traj_data/carla_true.pkl", "wb") as file:
        pickle.dump(traj_true_dict, file)
    #visualize_traj(traj_true_dict, 2, '1', [1, 2, 3, 4], legacy=True)

def run_perfect(data_path):
    traj_true = pd.read_csv(data_path)
    # Call the generator
    traj_Perfect = CP_data_generation(
        df=traj_true,
        gnn=None,
        Range=80.0,
        detect_mode='perfect',
    )

    traj_Perfect_dict = build_complete_trajectories_dict(traj_Perfect)
    with open("./CP_traj_data/carla_Perfect.pkl", "wb") as file:
        pickle.dump(traj_Perfect_dict, file)
    #visualize_traj(traj_Perfect_dict, 1, '1', [1, 2, 3, 4], legacy=True)

#%% carla RANDOM DROP detection
def run_randomdrop(data_path):

    traj_true = pd.read_csv(data_path)

    # Call the generator
    traj_Drop = CP_data_generation(
        df=traj_true,
        gnn=None,
        Range=80.0,
        bin_size=10.0,
        seed=42,
        detect_mode='random_drop',
    )


    traj_Drop_dict = build_complete_trajectories_dict(traj_Drop)
    with open("./CP_traj_data/carla_Drop.pkl", "wb") as file:
        pickle.dump(traj_Drop_dict, file)
    #visualize_traj(traj_Drop_dict, 1, '1', [1, 2, 3, 4], legacy=True)

def main():
    parser = argparse.ArgumentParser(
        description="Generate AV/CAV observations on trajectory dataset"
    )
    parser.add_argument(
        "--detection-mode",
        type=str,
        default="MIDAR",
        choices=["MIDAR","random_drop","perfect","true"],
        help="Which detection mode to generate CAV observations"
    )
    parser.add_argument("--use-ray-hit", action="store_true",
                        help="Use 5F [dist, ray_hit, w, l, h]. "
                             "If not set, use 4F [dist, w, l, h].")
    parser.add_argument("--occ-thresh", type=float, default=0.27)
    parser.add_argument(
        "--data-path",
        type=str,
        default="./traj_data/carla_vehicle_traj_1hz_local_interpolated_2_10000.csv",
        help="Path to input trajectory CSV file."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="../../trained_model/carla_los_graphormer_9385.pth",
        help="Path to trained LoS-Graphormer checkpoint."
    )

    args = parser.parse_args()

    if args.detection_mode == 'MIDAR':
        run_MIDAR(args.data_path, args.ckpt_path, args.occ_thresh, args.use_ray_hit)
    elif args.detection_mode == 'random_drop':
        run_randomdrop(args.data_path)
    elif args.detection_mode == 'perfect':
        run_perfect(args.data_path)
    elif args.detection_mode == 'true':
        run_true(args.data_path)

   
if __name__ == "__main__":
    main()
#python CP_data_generation.py --detection-mode MIDAR --ckpt-path ../../trained_model/carla_los_graphormer_4F_8982.pth --use-ray-hit --occ-thresh 0.27
#python CP_data_generation.py --detection-mode MIDAR --ckpt-path ../../trained_model/carla_los_graphormer_4F_8982.pth --occ-thresh 0.23
#python CP_data_generation.py --detection-mode 'true'
#python CP_data_generation.py --detection-mode 'random_drop'
#python CP_data_generation.py --detection-mode 'perfect'