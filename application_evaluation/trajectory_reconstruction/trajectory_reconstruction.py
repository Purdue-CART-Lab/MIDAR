#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 07:11:09 2025

@author: idiot
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:41:36 2025

@author: idiot
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
from pyomo.environ import *
import pyomo.environ as pyo
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.interpolate import griddata
from collections import defaultdict
import math
import argparse

def Preparation_for_optimization_indexed(all_veh_input, interval = 1):
    
    def identify_overtake_vehicles(df):
        """
        Identify vehicles that overtake (pass) at least one other, under the conditions that:
          1. Each vehicle starts and ends on the same lane.
          2. Overtaking is only checked within that same lane.
        Returns a sorted list of Vehicle_IDs that perform at least one overtake.
        """
    
        # ---------------------------------------------------------------
        # 1) Basic Preprocessing and 'Start/End Same Lane' Filter
        # ---------------------------------------------------------------
        df_sorted = df.sort_values(by=['Vehicle_ID', 'Global_Time'])
        
        # First and last records for each vehicle
        first_records = df_sorted.groupby('Vehicle_ID').first()
        last_records  = df_sorted.groupby('Vehicle_ID').last()
        
        # Vehicles whose first and last Lane_ID match
        same_lane_mask = (first_records['Lane_ID'] == last_records['Lane_ID'])
        vehicles_start_end_same = first_records.index[same_lane_mask]
    
        # OPTIONAL: If you also want to check that those vehicles visited >1 lane overall:
        df_detected = df_sorted[df_sorted['Detected'] == 1]
        lane_counts = df_detected.groupby('Vehicle_ID')['Lane_ID'].nunique()
        vehicles_start_end_same_multi = [
            v for v in vehicles_start_end_same
            if lane_counts.get(v, 0) > 1
        ]
        
        # Make a small dataframe for the start/end-same-lane subset
        start_end_same_times = pd.DataFrame({
            'Vehicle_ID': vehicles_start_end_same,
            'First_Time': first_records.loc[vehicles_start_end_same, 'Global_Time'],
            'Last_Time':  last_records.loc[vehicles_start_end_same, 'Global_Time'],
            'Start_Lane': first_records.loc[vehicles_start_end_same, 'Lane_ID'],
            'End_Lane':   last_records.loc[vehicles_start_end_same, 'Lane_ID']
        }).reset_index(drop=True)
        
        # Build a subset of the full dataframe containing ONLY these vehicles
        # (so that grouping or searching within each lane is faster)
        allowed_vehicles = set(vehicles_start_end_same)
        df_same = df_sorted[df_sorted['Vehicle_ID'].isin(allowed_vehicles)]
        
        # ---------------------------------------------------------------
        # 2) Group vehicles by *their* start/end lane
        #    (Because we only compare vehicles that share that same lane)
        # ---------------------------------------------------------------
        # Build a dict: lane -> list of vehicles whose (start_lane == end_lane == lane)
        lane_entries_exits = defaultdict(list)
        for row in start_end_same_times.itertuples(index=False):
            veh_id = row.Vehicle_ID
            lane   = row.Start_Lane
            
            # Subset for this one vehicle in this lane
            df_sub = df_same[(df_same['Vehicle_ID'] == veh_id) & (df_same['Lane_ID'] == lane)]
            if df_sub.empty:
                continue
            
            # ENTRY EVENT: the row with smallest (Global_Time, Local_Y)
            # nsmallest(1, columns=['Global_Time','Local_Y']) picks the earliest time,
            # breaking ties by the lowest localY.
            entry_row = df_sub.nsmallest(1, ['Global_Time','Local_Y']).iloc[0]
            entry_time    = entry_row['Global_Time']
            entry_local_y = entry_row['Local_Y']
            
            # EXIT EVENT: the row with largest (Global_Time, Local_Y)
            # nlargest(1, columns=['Global_Time','Local_Y']) picks the latest time,
            # breaking ties by the highest localY.
            exit_row = df_sub.nlargest(1, ['Global_Time','Local_Y']).iloc[0]
            exit_time    = exit_row['Global_Time']
            exit_local_y = exit_row['Local_Y']
            
            # Store
            lane_entries_exits[lane].append((
                veh_id,
                entry_time, entry_local_y,
                exit_time,  exit_local_y
            ))
        
        
        # ---------------------------------------------------------------
        # 3) Lane-by-lane Overtaking Check
        # ---------------------------------------------------------------
        overtakers = set()
    
        for lane, records in lane_entries_exits.items():
            if len(records) < 2:
                continue  # No comparisons if fewer than 2 vehicles in this lane
            
            # Convert to DataFrame for easier sorting
            lane_df = pd.DataFrame(
                records,
                columns=["Vehicle_ID", "entry_time", "entry_localY", "exit_time", "exit_localY"]
            )
            
            # ENTRY ORDER:
            #   Sort ascending by (entry_time, entry_localY)
            lane_df_entry_sorted = lane_df.sort_values(
                by=["entry_time", "entry_localY"],
                ascending=[True, True]
            ).reset_index(drop=True)
            
            # EXIT ORDER:
            #   Sort ascending by exit_time, but descending by exit_localY
            lane_df_exit_sorted = lane_df.sort_values(
                by=["exit_time", "exit_localY"],
                ascending=[True, False]
            ).reset_index(drop=True)
            
            # Build dicts to map each vehicle -> rank index
            exit_positions = {
                row.Vehicle_ID: i for i, row in lane_df_exit_sorted.iterrows()
            }
            
            # Compare pairs in ENTRY order
            vehicle_ids = lane_df_entry_sorted["Vehicle_ID"].tolist()
            n = len(vehicle_ids)
            
            for i in range(n):
                u = vehicle_ids[i]
                for j in range(i+1, n):
                    v = vehicle_ids[j]
                    
                    # By construction, v enters after u
                    # If v exits before u => v overtook u
                    if exit_positions[v] < exit_positions[u]:
                        overtakers.add(v)
        
        # ---------------------------------------------------------------
        # 4) Return or print final results
        # ---------------------------------------------------------------
        overtakers_sorted = sorted(overtakers)
        combined_set = vehicles_start_end_same_multi + overtakers_sorted
        final_vehicle_list = sorted(combined_set)
        
        return final_vehicle_list
    
    all_veh = all_veh_input.copy()
    detected_veh = all_veh[all_veh['Detected'] == 1]
    
    veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
    time_sequence = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
    time_sequence = [round(num, 1) for num in time_sequence]
    
    overtake_list = identify_overtake_vehicles(all_veh)
    overtake_list = [veh_id_list.index(veh_id) for veh_id in overtake_list]
    
    N=len(veh_id_list)#number of vehicles
    T=len(time_sequence)
    v_detected_max = max(detected_veh.v_Vel)
    
    start_dist={}
    start_v={}
    start_a={}
    start_lane={}
    arrival_time={}
    end_dist={}
    end_v={}
    end_lane={}
    departure_time={}
    veh_length={}
    for veh_id in veh_id_list:
        start_dist[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Local_Y'].iloc[0]
        start_v[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['v_Vel'].iloc[0]
        start_a[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['v_Acc'].iloc[0]
        arrival_time[veh_id_list.index(veh_id)]=time_sequence.index(all_veh[all_veh.Vehicle_ID==veh_id]['Global_Time'].iloc[0])
        end_dist[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Local_Y'].iloc[-1]
        end_v[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['v_Vel'].iloc[-1]
        departure_time[veh_id_list.index(veh_id)]=time_sequence.index(all_veh[all_veh.Vehicle_ID==veh_id]['Global_Time'].iloc[-1])
        start_lane[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Lane_ID'].iloc[0]
        end_lane[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Lane_ID'].iloc[-1]
        veh_len = all_veh[all_veh.Vehicle_ID==veh_id]['v_Length'].iloc[0]
        veh_length[veh_id_list.index(veh_id)] = min(veh_len, 4)
        
    detected_veh_id_list=detected_veh['Vehicle_ID'].drop_duplicates().tolist()
    for i in range(len(detected_veh)):
        idx = detected_veh.index[i]
        old_t = detected_veh.loc[idx, 'Global_Time']
        detected_veh.loc[idx, 'Global_Time'] = time_sequence.index(old_t)

    detected_dist={}
    detected_v={}
    detected_a={}
    detected_l={}
    for veh_id in detected_veh_id_list:
        detected_dist[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['Local_Y'].to_dict()
        detected_v[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['v_Vel'].to_dict()
        detected_a[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['v_Acc'].to_dict()
        detected_l[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['Lane_ID'].to_dict()
    
    
    detected_veh_id_list = [veh_id_list.index(item) for item in detected_veh_id_list]
    time_sequence=list(range(T))
    return N, T, start_dist, start_v, start_a, start_lane, arrival_time, end_dist, end_lane, end_v, departure_time, detected_dist, detected_v, detected_a, detected_l, detected_veh_id_list, v_detected_max, veh_length, overtake_list, time_sequence

def replay_results_complete(all_veh_list, gen_interval, case_index, date_id):
    
    opt_result=pd.read_csv('./results/'+gen_interval+'/'+str(case_index)+'/'+date_id+'.csv')

    all_veh=all_veh_list[case_index].copy()
    detected_veh = all_veh[all_veh['Detected'] == 1]
    
    veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
    
    time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
    time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
    for i in range(len(opt_result)):
        veh_id_temp = opt_result.Vehicle.iloc[i]
        time_id_temp = int(opt_result.Time.iloc[i])
        opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
        opt_result.loc[opt_result.index[i], 'Time']    = time_sequence_rp[time_id_temp]
    opt_result=opt_result.dropna(subset=['v'])
    
    for l in range(1,7):
        data_veh_l = opt_result[opt_result.l == l]
        all_veh_l = all_veh[all_veh.Lane_ID == l]
        detected_veh_l = detected_veh[detected_veh.Lane_ID == l]
        plt.figure(dpi=200)
        
        first_optimized = True
        first_ground_truth = True
        first_CAV = True
        first_Detected = True
        
        for veh_id in veh_id_list:
            data_veh_temp = data_veh_l[data_veh_l.Vehicle == veh_id]
            
            all_veh_temp = all_veh_l[all_veh_l.Vehicle_ID == veh_id]
            detected_veh_temp = detected_veh_l[detected_veh_l.Vehicle_ID == veh_id]
            
            if len(data_veh_temp) != 0:
                time_veh=list(data_veh_temp.Time)
                time_points=[[time_veh[0]]]
                for i in range(0,len(time_veh)-1):
                    if time_veh[i + 1] - time_veh[i] > 1:
                        time_points[-1].append(time_veh[i])
                        time_points.append([time_veh[i+1]])
                time_points[-1].append(time_veh[-1])
                for j in range(len(time_points)):
                    segment = data_veh_temp[(data_veh_temp.Time >= time_points[j][0]) & (data_veh_temp.Time <= time_points[j][1])]
                    if first_optimized:
                        plt.scatter(segment['Time'], segment['x'], color='green', s=8, label='Reconstructed')
                        plt.plot(segment['Time'], segment['x'], color='green',linewidth=2)
                        first_optimized = False
                    else:
                        plt.scatter(segment['Time'], segment['x'], color='green', s=8)
                        plt.plot(segment['Time'], segment['x'], color='green',linewidth=2)
            
            
            time_veh=list(all_veh_temp.Global_Time)
            if len(time_veh) == 0:
                continue
            time_points=[[time_veh[0]]]
            for i in range(0,len(time_veh)-1):
                if time_veh[i + 1] - time_veh[i] > 1:
                    time_points[-1].append(time_veh[i])
                    time_points.append([time_veh[i+1]])
            time_points[-1].append(time_veh[-1])
            
            for j in range(len(time_points)):
                segment = all_veh_temp[(all_veh_temp.Global_Time >= time_points[j][0]) & (all_veh_temp.Global_Time <= time_points[j][1])]
                if first_ground_truth:
                    plt.plot(segment['Global_Time'], segment['Local_Y'],color='chocolate', label='Undetected')
                    first_ground_truth = False
                else:
                    plt.plot(segment['Global_Time'], segment['Local_Y'],color='chocolate')      
            
            if len(detected_veh_temp) != 0:
                if (detected_veh_temp['CAV'] == 1).any():
                    color = 'red'
                else:
                    color = 'blue'
                time_veh_dtct=list(detected_veh_temp.Global_Time)
                time_points_dtct=[[time_veh_dtct[0]]]
                for i in range(0,len(time_veh_dtct)-1):
                    if time_veh_dtct[i + 1] - time_veh_dtct[i] > 1:
                        time_points_dtct[-1].append(time_veh_dtct[i])
                        time_points_dtct.append([time_veh_dtct[i+1]])
                time_points_dtct[-1].append(time_veh_dtct[-1])
    
                
                for j in range(len(time_points_dtct)):
                    segment = detected_veh_temp[(detected_veh_temp.Global_Time >= time_points_dtct[j][0]) & (detected_veh_temp.Global_Time <= time_points_dtct[j][1])]
                        
                    if color == 'red':
                        if first_CAV:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, label='CAV')
                            first_CAV = False
                        else:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color)
                    elif color == 'blue':
                        if first_Detected:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, label='Detected')
                            first_Detected = False
                        else:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color)
    
        plt.title('lane '+str(l))
        plt.xlim(min(opt_result.Time),max(opt_result.Time))
        plt.ylim(0,450)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (meters)')
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('./results/'+gen_interval+'/'+str(case_index)+'/'+date_id+'/'+str(l)+'.png')
        plt.close()
    
def optimization_process(all_veh_list_int, gen_interval, case_index, date_id):
    
    print('Case '+str(case_index)+' starts!')

    # Create the folder
    os.makedirs('./results/'+gen_interval+'/'+str(case_index), exist_ok=True)
    os.makedirs('./results/'+gen_interval+'/'+str(case_index)+'/'+date_id, exist_ok=True)
    
    N, T, start_dist, start_v, start_a, start_lane, arrival_time, end_dist, end_lane, end_v, departure_time, detected_dist, detected_v, detected_a, detected_l, detected_veh_id_list, v_detected_max, veh_length, overtake_list, time_sequence = Preparation_for_optimization_indexed(all_veh_list_int[case_index])

    rc_time = 0.3
    speed_limit = 33
    error_weight = 10
    x_weight = 0.00005
    a_weight = 0.1
    abslc_weight = 5
    mobil_weight = 0.05
    jerk_weight = 2
    jerk_low = -3
    jerk_high = 3
    
    #big M method
    M=10000

    # Create a Pyomo model
    model = ConcreteModel()

    model.V = RangeSet(0, N-1)  # Vehicles
    model.T = RangeSet(0, T-1)  # Time steps
    
    
    # Variables: acceleration of vehicle v at time t
    model.a = Var(model.V, model.T, domain=Reals)
    #absolute value of a
    model.abs_a = Var(model.V, model.T, domain=NonNegativeReals)
    # Absolute jerk of vehicle v at time t
    model.jerk_abs = Var(model.V, model.T, domain=NonNegativeReals)
    #absolute value of a
    model.lc_delta_a = Var(model.V, model.T, domain=Reals)
    # Velocity of vehicle v at time t
    model.v = Var(model.V, model.T, domain=NonNegativeReals)
    # Position of vehicle v at time t
    model.x = Var(model.V, model.T, domain=NonNegativeReals)
    # Lane_id of vehicle v at time t, 0, 1, 2, 3
    model.l = Var(model.V, model.T, domain=NonNegativeIntegers, bounds=(1, 6))
    # Lane changing decision of vehicle v at time t, -1, 0, 1
    model.lc = Var(model.V, model.T, domain=Integers, bounds=(-1, 1))
    #absolute value of lc
    model.abs_lc = Var(model.V, model.T, domain=Binary)
    # This will be indexed by (v, t).
    model.sign_lc = Var(model.V, model.T, within=Binary)
    # Variables indicating if vehicle v are in lane 0,1,2,3 at time t
    model.l1 = Var(model.V, model.T, within=Binary)
    model.l2 = Var(model.V, model.T, within=Binary)
    model.l3 = Var(model.V, model.T, within=Binary)
    model.l4 = Var(model.V, model.T, within=Binary)
    model.l5 = Var(model.V, model.T, within=Binary)
    model.l6 = Var(model.V, model.T, within=Binary)
    
    # Binary variable to decide if two vehicle are in the same lane
    model.z = Var(model.V, model.V, model.T, domain=Binary)

    # Auxilliary binary variable to decide if v is in front of u
    model.y = Var(model.V, model.V, model.T, domain=Binary)

    # Define auxiliary variables u and w
    model.u = Var(model.V, model.T, domain=NonNegativeReals)
    model.w = Var(model.V, model.T, domain=NonNegativeReals)

    # Objective: minimize the distance between detected and calculated positions, lane_id, minimize lane-changing times to avoid random change lane
    def objective_rule(model):
        return (
            error_weight * sum(
                model.u[v, t] + 3*model.w[v, t]
                for v in model.V 
                if v in detected_dist 
                for t in detected_dist[v]
            )
            + abslc_weight*sum(
                model.abs_lc[v, t]
                for t in time_sequence
                for v in model.V 
                if t >= arrival_time[v] and t <= departure_time[v]
            )
            + a_weight * sum(
                model.abs_a[v, t]
                for t in time_sequence
                for v in model.V 
                if t >= arrival_time[v] and t <= departure_time[v]
            )
            + jerk_weight * sum(
                model.jerk_abs[v, t]
                for t in time_sequence
                for v in model.V
                if t > arrival_time[v] and t <= departure_time[v]
            )
            - x_weight * sum(
                model.x[v, t]
                for t in time_sequence
                for v in model.V 
                if t >= arrival_time[v] and t <= departure_time[v]
            )- mobil_weight * sum(
                model.lc_delta_a[v,t] 
                for t in time_sequence
                for v in model.V 
                if t > arrival_time[v] and t <= departure_time[v]
            )
        )
    
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    # Constarints for objectoive to linearize the absolute values
    def abs_constraints_u1(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.u[v, t] >= model.x[v, t] - detected_dist[v][t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_u1 = Constraint(model.V, model.T, rule=abs_constraints_u1)

    def abs_constraints_u2(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.u[v, t] >= -(model.x[v, t] - detected_dist[v][t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_u2 = Constraint(model.V, model.T, rule=abs_constraints_u2)

    def abs_constraints_w1(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.w[v, t] >= model.l[v, t] - detected_l[v][t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_w1 = Constraint(model.V, model.T, rule=abs_constraints_w1)

    def abs_constraints_w2(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.w[v, t] >= -(model.l[v, t] - detected_l[v][t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_w2 = Constraint(model.V, model.T, rule=abs_constraints_w2)

    
    # Distance, Speed, and Acceleration
    def start_dist_rule(model, v):
        return model.x[v, arrival_time[v]] == start_dist[v]
    model.start_dist = Constraint(model.V, rule=start_dist_rule)
    
    def start_v_rule(model, v):
        return model.v[v, arrival_time[v]] == start_v[v]
    model.start_v = Constraint(model.V, rule=start_v_rule)
    
    def distance_end_rule(model, v, t):
        if t == departure_time[v]:
            return model.x[v, t] == end_dist[v]
        else:
            return Constraint.Skip
    model.distance_end = Constraint(model.V, model.T, rule=distance_end_rule)
    
    def distance_update_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v, t] == model.x[v, t-1] + model.v[v, t-1] + 0.5 * model.a[v, t-1]
        else:
            return Constraint.Skip
    model.distance_update = Constraint(model.V, model.T, rule=distance_update_rule)
    
    def velocity_update_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.v[v, t] == model.v[v, t-1] + model.a[v, t-1]
        else:
            return Constraint.Skip
    model.velocity_update = Constraint(model.V, model.T, rule=velocity_update_rule)
    
    def velocity_limit_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.v[v, t] <= speed_limit  # Use '<=' instead of '<'
        else:
            return Constraint.Skip
    model.velocity_limit = Constraint(model.V, model.T, rule=velocity_limit_rule)
    
    def acc_limit_rule(model,v,t):
        if t >= arrival_time[v] and t <= departure_time[v]:
            return (-3.5, model.a[v, t], 3.5) # first try (-3.5,3.5)
        else:
            return Constraint.Skip
    model.acc_limit = Constraint(model.V, model.T, rule=acc_limit_rule)
    
    def abs_a_constraints_1(model, v, t):
        return model.abs_a[v, t] >= model.a[v, t]
    model.abs_a_constraints_1 = Constraint(model.V, model.T, rule=abs_a_constraints_1)

    def abs_a_constraints_2(model, v, t):
        return model.abs_a[v, t] >= -model.a[v, t]
    model.abs_a_constraints_2 = Constraint(model.V, model.T, rule=abs_a_constraints_2)
    
    def jerk_rule(model,v,t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return (jerk_low, model.a[v, t] - model.a[v, t-1], jerk_high) #-3 3
        else:
            return Constraint.Skip
    model.jerk = Constraint(model.V, model.T, rule=jerk_rule)
      
    def jerk_abs_1_rule(model, v, t):
        # Only relevant if t> arrival_time[v] (so that t-1 is valid)
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return model.jerk_abs[v, t] >= model.a[v, t] - model.a[v, t-1]
        else:
            return Constraint.Skip
    model.jerk_abs_1 = Constraint(model.V, model.T, rule=jerk_abs_1_rule)
    
    def jerk_abs_2_rule(model, v, t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return model.jerk_abs[v, t] >= -(model.a[v, t] - model.a[v, t-1])
        else:
            return Constraint.Skip
    model.jerk_abs_2 = Constraint(model.V, model.T, rule=jerk_abs_2_rule)
    
    def start_lane_rule(model, v):
        return model.l[v, arrival_time[v]] == start_lane[v]
    model.start_lane = Constraint(model.V, rule=start_lane_rule)

    def end_lane_rule(model, v):
        return model.l[v, departure_time[v]] == end_lane[v]
    model.end_lane = Constraint(model.V, rule=end_lane_rule)

    def l_update_rule(model, v, t):
        if t>arrival_time[v] and t <= departure_time[v]:
            return model.l[v,t] == model.l[v,t-1] + model.lc[v,t]
        else:
            return Constraint.Skip
    model.l_update = Constraint(model.V, model.T, rule=l_update_rule)
    
    def no_lane_changing_l_rule(model,v,t):
        if start_lane[v]==end_lane[v] and v not in overtake_list:
            if t >= arrival_time[v] and t <= departure_time[v]:
                return model.l[v, t] == model.l[v, arrival_time[v]]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.no_lane_changing_l = Constraint(model.V, model.T, rule=no_lane_changing_l_rule)

    def no_lane_changing_lc_rule(model,v,t):
        if start_lane[v]==end_lane[v] and v not in overtake_list:
            if t > arrival_time[v] and t <= departure_time[v]:
                return model.lc[v, t] == 0
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.no_lane_changing_lc = Constraint(model.V, model.T, rule=no_lane_changing_lc_rule)
    
    def lc_direction_rule(model, v, t):
        return sum(model.lc[v, t] for t in time_sequence) == end_lane[v] - start_lane[v]
    model.lc_direction = Constraint(model.V, model.T, rule=lc_direction_rule)

    # (A) Enforce sign logic for lc[v,t].
    #     If sign_lc[v,t] = 1 ==> lc[v,t] >= 0
    #     If sign_lc[v,t] = 0 ==> lc[v,t] <= 0
    def sign_lc_1_rule(model, v, t):
        # lc[v,t] <= M * sign_lc[v,t]
        return model.lc[v, t] <= M * model.sign_lc[v, t]
    model.sign_lc_1 = Constraint(model.V, model.T, rule=sign_lc_1_rule)
    
    def sign_lc_2_rule(model, v, t):
        # lc[v,t] >= -M * (1 - sign_lc[v,t])
        return model.lc[v, t] >= -M * (1 - model.sign_lc[v, t])
    model.sign_lc_2 = Constraint(model.V, model.T, rule=sign_lc_2_rule)
    
    # (B) Enforce abs_lc[v,t] = |lc[v,t]|
    def abs_lc_1_rule(model, v, t):
        return model.abs_lc[v, t] >= model.lc[v, t]
    model.abs_lc_1 = Constraint(model.V, model.T, rule=abs_lc_1_rule)
    
    def abs_lc_2_rule(model, v, t):
        return model.abs_lc[v, t] >= -model.lc[v, t]
    model.abs_lc_2 = Constraint(model.V, model.T, rule=abs_lc_2_rule)
    
    def abs_lc_3_rule(model, v, t):
        # abs_lc[v,t] <= lc[v,t] + M*(1 - sign_lc[v,t])
        return model.abs_lc[v, t] <= model.lc[v, t] + M * (1 - model.sign_lc[v, t])
    model.abs_lc_3 = Constraint(model.V, model.T, rule=abs_lc_3_rule)
    
    def abs_lc_4_rule(model, v, t):
        # abs_lc[v,t] <= -lc[v,t] + M*sign_lc[v,t]
        return model.abs_lc[v, t] <= -model.lc[v, t] + M * model.sign_lc[v, t]
    model.abs_lc_4 = Constraint(model.V, model.T, rule=abs_lc_4_rule)
    
    
    def lc_delta_a_upper_rule(model, v, t):
        # lc_delta_a[v,t] ≤ (a[v,t] - a[v,t-1]) + M*(1 - abs_lc[v,t])
        # meaning if abs_lc[v,t] = 1 => lc_delta_a[v,t] ≤ a[v,t] - a[v,t-1]
        # if abs_lc[v,t] = 0 => no binding from this constraint
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t]
                    <= (model.a[v,t] - model.a[v,t-1])
                    + M*(1 - model.abs_lc[v,t]))
        else:
            return Constraint.Skip 
    model.lc_delta_a_upper = Constraint(model.V, model.T, rule=lc_delta_a_upper_rule)
    
    def lc_delta_a_lower_rule(model, v, t):
        # lc_delta_a[v,t] ≥ (a[v,t] - a[v,t-1]) - M*(1 - abs_lc[v,t])
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t]
                    >= (model.a[v,t] - model.a[v,t-1])
                    - M*(1 - model.abs_lc[v,t]))
        else:
            return Constraint.Skip
    model.lc_delta_a_lower = Constraint(model.V, model.T, rule=lc_delta_a_lower_rule)
    
    def lc_delta_a_zero_upper_rule(model, v, t):
        # If no lane change => lc_delta_a[v,t] ≤ +M * abs_lc[v,t]
        # i.e. if abs_lc[v,t] = 0 => lc_delta_a[v,t] ≤ 0
        # combined with the next constraint => 0
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t] 
                    <= M * model.abs_lc[v,t])
        else:
            return Constraint.Skip
    model.lc_delta_a_zero_upper = Constraint(model.V, model.T, rule=lc_delta_a_zero_upper_rule)
    
    def lc_delta_a_zero_lower_rule(model, v, t):
        # If no lane change => lc_delta_a[v,t] ≥ -M * abs_lc[v,t]
        # i.e. if abs_lc[v,t] = 0 => lc_delta_a[v,t] ≥ 0
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t]
                    >= -M * model.abs_lc[v,t])
        else:
            return Constraint.Skip
    model.lc_delta_a_zero_lower = Constraint(model.V, model.T, rule=lc_delta_a_zero_lower_rule)
    
    #Create model.l0/l1/l2/l3 variable, model.l0=1 means vehicle v is in lane 0 at time t
    def l_binary_1_rule(model, v, t):
        if arrival_time[v] <= t <= departure_time[v]:
            return (
                model.l1[v, t]
              + model.l2[v, t]
              + model.l3[v, t]
              + model.l4[v, t]
              + model.l5[v, t]
              + model.l6[v, t]
            ) == 1
        else:
            return Constraint.Skip
    model.l_binary_1 = Constraint(model.V, model.T, rule=l_binary_1_rule)
    
    def l_binary_2_rule(model, v, t):
        if arrival_time[v] <= t <= departure_time[v]:
            return model.l[v, t] == (
                1*model.l1[v, t] +
                2*model.l2[v, t] +
                3*model.l3[v, t] +
                4*model.l4[v, t] +
                5*model.l5[v, t] +
                6*model.l6[v, t]
            )
        else:
            return Constraint.Skip
    model.l_binary_2 = Constraint(model.V, model.T, rule=l_binary_2_rule)
    
    def y_constraint_rule_1(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[v,t] - model.x[u,t] >= 1e-20- M * (1 - model.y[v, u, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    
    model.y_constraint_1 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_1)
    
    
    def y_constraint_rule_2(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[v,t] - model.x[u,t] <=  M * model.y[v, u, t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    
    model.y_constraint_2 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_2)
    
    #if two vehicles are in the same lane and don't make LC at the current lane, then their lanes at the next time stamp should be the same
    def y_constraint_rule_3(model, v, u, t):
        if v != u:
            if t > arrival_time[v] and t <= departure_time[v] and t > arrival_time[u] and t <= departure_time[u]:
                return model.y[v, u, t] - model.y[v, u, t-1] >= -M * (model.abs_lc[v,t] + model.abs_lc[u,t] + model.z[v,u,t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    
    model.y_constraint_3 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_3)
    
    def y_constraint_rule_4(model, v, u, t):
        if v != u:
            if t > arrival_time[v] and t <= departure_time[v] and t > arrival_time[u] and t <= departure_time[u]:
                return model.y[v, u, t] - model.y[v, u, t-1] <= M * (model.abs_lc[v,t] + model.abs_lc[u,t] + model.z[v,u,t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    
    model.y_constraint_4 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_4)
    
    # --- Lane 1 (l1[v,t]) ---
    def z_constraint_rule_1(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l1[u,t] - model.l1[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_2(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l1[v,t] - model.l1[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_3(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            # "2 - l1[v,t] - l1[u,t]" ensures z[v,u,t] can be zero if v and u are in the same lane
            return model.z[v,u,t] <= 2 - model.l1[v,t] - model.l1[u,t]
        return Constraint.Skip
    
    # --- Lane 2 (l2[v,t]) ---
    def z_constraint_rule_4(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l2[u,t] - model.l2[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_5(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l2[v,t] - model.l2[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_6(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l2[v,t] - model.l2[u,t]
        return Constraint.Skip
    
    # --- Lane 3 (l3[v,t]) ---
    def z_constraint_rule_7(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l3[u,t] - model.l3[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_8(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l3[v,t] - model.l3[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_9(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l3[v,t] - model.l3[u,t]
        return Constraint.Skip
    
    # --- Lane 4 (l4[v,t]) ---
    def z_constraint_rule_10(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l4[u,t] - model.l4[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_11(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l4[v,t] - model.l4[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_12(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l4[v,t] - model.l4[u,t]
        return Constraint.Skip
    
    # --- Lane 5 (l5[v,t]) ---
    def z_constraint_rule_13(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l5[u,t] - model.l5[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_14(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l5[v,t] - model.l5[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_15(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l5[v,t] - model.l5[u,t]
        return Constraint.Skip
    
    # --- Lane 6 (l6[v,t]) ---
    def z_constraint_rule_16(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l6[u,t] - model.l6[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_17(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l6[v,t] - model.l6[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_18(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l6[v,t] - model.l6[u,t]
        return Constraint.Skip

    
    # ---------------------------------------------------------------------
    # ADD z-CONSTRAINTS TO THE MODEL
    # ---------------------------------------------------------------------
    model.z_constraint_1  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_1)
    model.z_constraint_2  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_2)
    model.z_constraint_3  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_3)
    model.z_constraint_4  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_4)
    model.z_constraint_5  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_5)
    model.z_constraint_6  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_6)
    model.z_constraint_7  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_7)
    model.z_constraint_8  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_8)
    model.z_constraint_9  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_9)
    model.z_constraint_10 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_10)
    model.z_constraint_11 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_11)
    model.z_constraint_12 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_12)
    model.z_constraint_13 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_13)
    model.z_constraint_14 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_14)
    model.z_constraint_15 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_15)
    model.z_constraint_16 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_16)
    model.z_constraint_17 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_17)
    model.z_constraint_18 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_18)
    
    
    ##############################################################################
    # 1) Force x[v,t] in [200,400] if abs_lc[v,t]==1 AND l6[v,t]==1
    ##############################################################################
    
    def lane6_change_now_x_lower_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t]=1 => x[v,t]>=200
        # otherwise relaxed by big-M
        if t >= arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] >= 187 - M * (2 - model.abs_lc[v,t] - model.l6[v,t])
        else:
            return Constraint.Skip
    model.lane6_change_now_x_lower = Constraint(model.V, model.T, rule=lane6_change_now_x_lower_rule)
    
    def lane6_change_now_x_upper_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t]=1 => x[v,t]<=400
        if t >= arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] <= 400 + M * (2 - model.abs_lc[v,t] - model.l6[v,t])
        else:
            return Constraint.Skip
    model.lane6_change_now_x_upper = Constraint(model.V, model.T, rule=lane6_change_now_x_upper_rule)
    
    
    ##############################################################################
    # 2) Force x[v,t] in [200,400] if abs_lc[v,t]==1 AND l6[v,t-1]==1
    ##############################################################################
    
    def lane6_change_prev_x_lower_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t-1]=1 => x[v,t]>=200
        # Make sure t>arrival_time[v] so t-1 is valid
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] >= 187 - M * (2 - model.abs_lc[v,t] - model.l6[v,t-1])
        else:
            return Constraint.Skip
    model.lane6_change_prev_x_lower = Constraint(model.V, model.T, rule=lane6_change_prev_x_lower_rule)
    
    def lane6_change_prev_x_upper_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t-1]=1 => x[v,t]<=400
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] <= 400 + M * (2 - model.abs_lc[v,t] - model.l6[v,t-1])
        else:
            return Constraint.Skip
    model.lane6_change_prev_x_upper = Constraint(model.V, model.T, rule=lane6_change_prev_x_upper_rule)

    
    #if model.y[v, u, t]=1, model.x[v,t]>model.x[u,t], v is in front of u
    #if model.z[v, u, t]=0, two vehicles are on the same lane
    def car_following_1_rule(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[v,t] - model.x[u,t] >= veh_length[v] + rc_time*model.v[u,t] - M * model.z[v, u, t] - M * (1 - model.y[v, u, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.car_following_1 = Constraint(model.V, model.V, model.T, rule=car_following_1_rule)
    
    def car_following_2_rule(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[u,t] - model.x[v,t] >= veh_length[u] + rc_time*model.v[v,t] - M * model.z[v, u, t] - M * model.y[v, u, t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.car_following_2 = Constraint(model.V, model.V, model.T, rule=car_following_2_rule)
    
    
    # Solve the model
    solver = SolverFactory("gurobi_direct")#SolverFactory('cbc')
    
    
    start_time = time.time()  # Start timer
    results = solver.solve(model)
    end_time = time.time()  # End timer

    optimization_time = end_time - start_time
    #print(optimization_time)

    # Open the file in write mode and save the text
    with open('./results/'+gen_interval+'/'+str(case_index)+'/'+date_id+'/time.txt', 'w') as file:
        file.write(str(optimization_time))
    
    
    def extract_variables_to_dataframe(model, variable_name):
        var = getattr(model, variable_name)
        data = []
    
        for v in model.V:
            for t in model.T:
                # If (v, t) is not a valid index for this Var, record None
                if (v, t) not in var:
                    data.append((v, t, None))
                    continue
    
                # Safe access: .value might be None if unsolved/uninitialized
                val = var[v, t].value
                data.append((v, t, val))
    
        df = pd.DataFrame(data, columns=['Vehicle', 'Time', variable_name])
        return df
    
    a_df = extract_variables_to_dataframe(model, 'a')
    v_df = extract_variables_to_dataframe(model, 'v')
    x_df = extract_variables_to_dataframe(model, 'x')
    l_df = extract_variables_to_dataframe(model, 'l')
    lc_df = extract_variables_to_dataframe(model, 'lc')
    abs_lc_df = extract_variables_to_dataframe(model, 'abs_lc')
    l1_df = extract_variables_to_dataframe(model, 'l1')
    l2_df = extract_variables_to_dataframe(model, 'l2')
    l3_df = extract_variables_to_dataframe(model, 'l3')
    l4_df = extract_variables_to_dataframe(model, 'l4')
    l5_df = extract_variables_to_dataframe(model, 'l5')
    l6_df = extract_variables_to_dataframe(model, 'l6')
    
    dfs = [a_df, v_df, x_df, l_df, lc_df, abs_lc_df, l1_df, l2_df, l3_df,l4_df, l5_df, l6_df]#, x_nlc_df, v_nlc_df]

    # Merging dataframes on 'vehicle' and 'time'
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['Vehicle', 'Time'])

    csv_file_path = './results/'+gen_interval+'/'+str(case_index)+'/'+date_id+'.csv'
    merged_df.to_csv(csv_file_path, index=False)
    
    replay_results_complete(all_veh_list_int, gen_interval, case_index, date_id)
    

#%%MAE
def MAE_MAPE_calculation_all(all_veh_list_int, gen_interval, date_id, case_number):
    columns = ['Vehicle_ID', 'Global_Time', 'x', 'Local_Y','l','Lane_ID']#,'l','Lane_ID'
    final_df = pd.DataFrame(columns=columns)
    mae_list = []
    mape_list = []
    rmse_list = []
    mae_l_list = []
    
    for case_index in range(1, case_number + 1):
        # You can filter out unwanted cases in the following if-statement if needed.
        if case_index not in [22,23,33,43]:  # For example, if you want to skip case 7, 16, 36, add: if case_index not in [7,16,36]:
            # Read the prediction result for the current case
            opt_result = pd.read_csv('./results/{}/{}/{}.csv'.format(gen_interval, case_index, date_id))
            print(case_index)
            
            # Get the corresponding vehicle data for the case (assumes all_veh_list_int is defined globally)
            all_veh = all_veh_list_int[case_index].copy()
            veh_id_list = all_veh['Vehicle_ID'].drop_duplicates().tolist()
            
            # Create a time sequence rounded to one decimal place
            time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time) + 1, 1).tolist()
            time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
            
            # Replace the integer indices in the prediction results with the actual vehicle IDs and times
            for i in range(len(opt_result)):
                veh_id_temp = opt_result.Vehicle.iloc[i]
                time_id_temp = int(opt_result.Time.iloc[i])
                opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
                opt_result.loc[opt_result.index[i], 'Time'] = time_sequence_rp[time_id_temp]
            
            # Drop rows with missing predictions (assuming column 'v' holds a value you care about)
            opt_result = opt_result.dropna(subset=['v'])
            
            # Rename columns to prepare for merging
            opt_result.rename(columns={'Vehicle': 'Vehicle_ID', 'Time': 'Global_Time'}, inplace=True)
            
            # Merge the prediction results with the actual vehicle data
            merged_df = pd.merge(opt_result, all_veh, on=['Vehicle_ID', 'Global_Time'])
            
            # Calculate MAE for the current case and store it
            mae_val = mean_absolute_error(merged_df['x'], merged_df['Local_Y'])
            mae_list.append(mae_val)
            
            # Calculate MAPE for the current case.
            # Using the formula: MAPE = mean(|(true - pred) / true|) * 100.
            mape_val = np.mean(np.abs((merged_df['x'] - merged_df['Local_Y']) / merged_df['Local_Y'])) * 100
            mape_list.append(mape_val)
            
            # Calculate RMSE for the current case
            rmse_val = math.sqrt(mean_squared_error(merged_df['x'], merged_df['Local_Y']))
            rmse_list.append(rmse_val)
            
            mae_l = mean_absolute_error(merged_df['l'], merged_df['Lane_ID'])
            mae_l_list.append(mae_l)
            
            # Keep only the desired columns and append to the final dataframe.
            merged_df = merged_df[columns]
            final_df = pd.concat([final_df, merged_df], ignore_index=True)
    
    # Calculate overall MAE and overall MAPE across all cases
    overall_mae = mean_absolute_error(final_df['x'], final_df['Local_Y'])
    overall_mape = np.mean(np.abs((final_df['x'] - final_df['Local_Y']) / final_df['Local_Y'])) * 100
    overall_rmse = math.sqrt(mean_squared_error(final_df['x'], final_df['Local_Y']))
    overall_mae_l = mean_absolute_error(final_df['l'], final_df['Lane_ID'])

    return overall_mae, mae_list, overall_mape, mape_list, overall_rmse, rmse_list, overall_mae_l, mae_l_list

def lane_changing_timing(all_veh_list_int, gen_interval, date_id, case_number):
    
    lc_time_diff = []
    count = 0
    lc_time_diff_dtct = []
    count_dtct = 0
    count_error = 0
    
    for case_index in range(1, case_number+1):
        print(case_index)
        if case_index not in [22,23,33,43]:
            
            #case_index=3
            
            opt_result=pd.read_csv('./results/'+gen_interval+'/'+str(case_index)+'/'+date_id+'.csv')
            all_veh=all_veh_list_int[case_index].copy()
            
            veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
            
            time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
            time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
            for i in range(len(opt_result)):
                veh_id_temp = opt_result.Vehicle.iloc[i]
                time_id_temp = int(opt_result.Time.iloc[i])
                opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
                opt_result.loc[opt_result.index[i], 'Time']    = time_sequence_rp[time_id_temp]
            opt_result=opt_result.dropna(subset=['v'])
            opt_result.rename(columns={'Vehicle': 'Vehicle_ID'}, inplace=True)
            opt_result.rename(columns={'Time': 'Global_Time'}, inplace=True)
            
            merged_df = pd.merge(opt_result,  all_veh, on=['Vehicle_ID', 'Global_Time'])
            
            lane_GT={}
            lane_opt={}
            lane_GT_dtct={}
            for veh_id in veh_id_list:
                lane_GT[veh_id]={}
                lane_opt[veh_id]={}
                lane_GT_dtct[veh_id]={}
                veh_temp = merged_df[merged_df.Vehicle_ID == veh_id]
                for j in range(len(veh_temp)-1):
                    if veh_temp.Lane_ID.iloc[j] != veh_temp.Lane_ID.iloc[j+1]:
                        lane_GT[veh_id][(veh_temp.Lane_ID.iloc[j],veh_temp.Lane_ID.iloc[j+1])] = veh_temp.Global_Time.iloc[j+1]
                        if veh_temp.Detected.iloc[j+1] == 0 or veh_temp.Detected.iloc[j] == 0:
                            lane_GT_dtct[veh_id][(veh_temp.Lane_ID.iloc[j],veh_temp.Lane_ID.iloc[j+1])] = veh_temp.Global_Time.iloc[j+1]
                    if veh_temp.l.iloc[j] != veh_temp.l.iloc[j+1]:
                        lane_opt[veh_id][(veh_temp.l.iloc[j],veh_temp.l.iloc[j+1])] = veh_temp.Global_Time.iloc[j+1]
            
            for k, v in lane_GT.items():
                if len(v) == 0:
                    continue
                else:
                    for k_sub, v_sub in v.items():
                        if k_sub in lane_opt[k]:
                            lc_time_diff.append(abs(v_sub-lane_opt[k][k_sub]))
                            count+=1
                            '''
                            if abs(v_sub-lane_opt[k][k_sub]) == 14:
                                error=[case_index,lane_opt[k][k_sub],k,k_sub]
                                '''
                        else:
                            count_error+=1
            
            for k, v in lane_GT_dtct.items():
                if len(v) == 0:
                    continue
                else:
                    for k_sub, v_sub in v.items():
                        if k_sub in lane_opt[k]:
                            lc_time_diff_dtct.append(abs(v_sub-lane_opt[k][k_sub]))
                            count_dtct+=1

                        
    return lc_time_diff,count,lc_time_diff_dtct,count_dtct,count_error

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory reconstruction evaluation (MIDAR vs baselines)."
    )

    # core inputs
    parser.add_argument(
        "--pkl-path",
        type=str,
        default="carla_MIDAR.pkl",
        help="Path to trajectory reconstruction pickle file "
             "(dict[case_id -> DataFrame]).",
    )
    parser.add_argument(
        "--occthr",
        type=float,
        default=0.27,
        help="the threshold to determine if a detection is occluded (False Negative)"
    )

    args = parser.parse_args()

    with open(args.pkl_path, "rb") as file:
        all_veh_list_int = pickle.load(file)

    recon_num = 0
    all_num = 0
    for i,df in all_veh_list_int.items():
        recon_num += len(df[df.Detected == 0])
        recon_num -= len(set(df.Vehicle_ID))*2
        all_num += len(df)
    recon_rate = recon_num/all_num
    print('recon_rate: ', recon_rate)

    base_path = os.path.basename(args.pkl_path)            # "carla_MIDAR.pkl"
    path = os.path.splitext(base_path)[0]         # "carla_MIDAR"
    occthr = str(args.occthr)
    case_number = len(all_veh_list_int)
    
    for i in range(1, case_number+1):
        if i in [22,23,33,43]: # contains errouneous data
            continue
        else:
            try:
                optimization_process(all_veh_list_int, path, i, occthr)
            except Exception as e:
                print(f"Error occurred at i={i}: {e}")
                pass
                
    overall_mae, mae_list, overall_mape, mape_list, overall_rmse, rmse_list, overall_mae_l, mae_l_list = MAE_MAPE_calculation_all(all_veh_list_int, path,occthr,case_number)
    lc_time_diff,count,lc_time_diff_dtct,count_dtct,count_err = lane_changing_timing(all_veh_list_int, path,occthr,case_number)

    print('mae_x: ',overall_mae)
    print('mape_x: ',overall_mape)
    print('rmse_x: ',overall_rmse)
    print('mae_k: ',overall_mae_l)
    print('mae_lc: ',sum(lc_time_diff)/count)

if __name__ == "__main__":
    main()

# python trajectory_reconstruction.py --pkl-path ./CP_traj_data/carla_MIDAR.pkl --occthr 0.27
# python trajectory_reconstruction.py --pkl-path ./CP_traj_data/carla_MIDAR_025.pkl --occthr 0.25