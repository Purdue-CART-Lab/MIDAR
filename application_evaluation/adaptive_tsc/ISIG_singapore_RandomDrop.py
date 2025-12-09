#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:43:56 2025

@author: idiot
"""

# import packages
import math
import numpy as np
import random
import traci
import os
import sys

def find_last_effective_element(veh_routes):
    for element in reversed(veh_routes):
        if not element.startswith(":"):
            return element


def mapping_route2phase(veh_routes):
    lane_id = find_last_effective_element(veh_routes)
    if lane_id in ['74859235#1_2','74859235#1_1','652556221_2']:  # south bound start
        return 0
    elif lane_id in ['74859235#1_0','744913575_1']:
        return 1
    elif lane_id in ['744913575_0',]:
        return 2
    elif lane_id in ['652556221_1']:
        if random.random() <= 0.5:
            return 0
        else:
            return 1
    elif lane_id in ['652556221_0','652556226_0']:
        if random.random() <= 0.5:
            return 1
        else:
            return 2
    elif lane_id in ['652556226_1']:
        if random.random() <= 0.67:
            return 0
        else:
            return 1
    elif lane_id in ['840414102_4','840414102_3','840414102_2','652291042_3','652291042_2','652291042_1','660362915_2','660362915_1']:  # west bound start
        return 3
    elif lane_id in ['840414102_1','840414102_0','652291042_0','660362915_0']:
        return 4
    elif lane_id in ['840414101_4','840414101_3','479616091_3','479616091_2','654999305_2']:
        return 5
    elif lane_id in ['840414101_2','840414101_1','479616091_1','654999305_1']:
        return 6
    elif lane_id in ['840414101_0']:
        return 7
    elif lane_id in ['479616091_0','654999305_0']:
        if random.random() <= 0.5:
            return 6
        else:
            return 7
    elif lane_id in ['173767662_5','173767662_4','654985989_4']:
        return 8
    elif lane_id in ['173767662_3','173767662_2','173767662_1','654985989_3','654985989_2','654985989_1','630390098_1','630390098_2']:
        return 9
    elif lane_id in ['173767662_0','654985989_0','630390098_0']:
        return 10
    elif lane_id in ['630390098_3']:
        if random.random() <= 0.67:
            return 8
        else:
            return 9


def phase2ETA(veh_id, intersection_center):
    veh_speed = traci.vehicle.getSpeed(veh_id)
    x2, y2 = intersection_center
    if veh_speed <= 0.9:
        return 0
    else:
        x1, y1 = traci.vehicle.getPosition(veh_id)
        return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)/veh_speed)


def return_ETA_cell(veh_id, intersection_center, plan_horizon, routes):
    phase_id = -1  # False bug ?idk
    phase_id = mapping_route2phase(routes[veh_id])
    ETA = phase2ETA(veh_id, intersection_center)
    if ETA >= plan_horizon-1:
        return False
    else:
        if phase_id != None:
            return [ETA, phase_id]

# DP-related functions
def return_signal(phase):  # input '16','15'...
    if phase == "012":
        return [6,7,8]  # phase_id in SUMO
    elif phase == '34':
        return [3, 4, 5]
    elif phase == '567':
        return [9,10,11]
    elif phase == '8910':
        return [0,1,2]


# given state variable and decision variable, calculate value function
def signal_DP_sup_2(s, x, phase, prev_stage_calculation, G_min_T, Arrival_Table, offset):
    # backup G_min_T[phase_sequence[phase_sequence.index(phase)-1]]
    prev_stage_bu = prev_stage_calculation.copy()[s-x-(Yellow+Red)*2-offset]
    phase_loc_1, phase_loc_2, phase_loc_3 = return_phase_loc_spec(phase)
    # clearance
    for i in range(s-5-x+1, s-5+1):
        prev_stage_bu = np.vstack((prev_stage_bu, prev_stage_bu[i-1,]+np.append(Arrival_Table[i, :], [0, 0])))
        prev_stage_bu[i, phase_loc_1] = np.maximum(prev_stage_bu[i, phase_loc_1] - departure_rate, 0)
        prev_stage_bu[i, phase_loc_2] = np.maximum(prev_stage_bu[i, phase_loc_2] - departure_rate*2, 0)
        prev_stage_bu[i, phase_loc_3] = np.maximum(prev_stage_bu[i, phase_loc_3] - departure_rate*3, 0)
        prev_stage_bu[i, -1] = prev_stage_bu[i-1, -1] + np.sum(prev_stage_bu[i, :approaches])
    for i in range(s-5+1, s+1):
        prev_stage_bu = np.vstack((prev_stage_bu, prev_stage_bu[i-1,]+np.append(Arrival_Table[i, :], [0, 0])))
        prev_stage_bu[i, -1] = prev_stage_bu[i-1, -1] + np.sum(prev_stage_bu[i, :approaches])
    return x, prev_stage_bu[-1, -1], prev_stage_bu

# given state variable, choose the decision variable value that minimize the value function
def signal_DP_sup_1(s, phase, prev_stage_calculation, G_min_T, G_max_T, Arrival_Table, s_start, offset):
    if s-(G_min_T[phase]+Yellow+Red) < s_start+1:
        flag = True
    else:
        flag = False
    upload = [-1, np.inf]
    X = []
    for i in range(G_min_T[phase], G_max_T+1):
        if s-5-i >= s_start-Yellow-Red-G_min_T[phase]:
            X.append(i)
    for x_minor in X:
        x, value, stage = signal_DP_sup_2(s, x_minor, phase, prev_stage_calculation, G_min_T, Arrival_Table, offset)
        if value < upload[1]: # < instead of <=
            upload = [x, value, stage, flag]
    return upload

def return_phase_loc(phase):
    if phase == '012':
        phase_loc = [0,1,2]
    elif phase == '34':
        phase_loc = [3,4]
    elif phase == '567':
        phase_loc = [5,6,7]
    elif phase == '8910':
        phase_loc = [8,9,10]
    return phase_loc

def return_phase_loc_spec(phase):
    if phase == '012':
        phase_loc_1 = [2]
        phase_loc_2 = [0,1]
        phase_loc_3 = []
    elif phase == '34':
        phase_loc_1 = []
        phase_loc_2 = [4]
        phase_loc_3 = [3]
    elif phase == '567':
        phase_loc_1 = [7]
        phase_loc_2 = [5,6]
        phase_loc_3 = []
    elif phase == '8910':
        phase_loc_1 = [10]
        phase_loc_2 = [8]
        phase_loc_3 = [9]
    return phase_loc_1, phase_loc_2, phase_loc_3

def return_phase_char(index):
    if index == 0:
        return '8910'
    elif index == 3:
        return '34'
    elif index == 6:
        return '012'
    elif index == 9:
        return '567'

# given arrival table and other predefined parameters, obtain the lowest value function and the best decision at every stage and every timestamp
def signal_DP(Arrival_Table, G_min_T, G_max_T, phase_sequence, plan_horizon, approaches, departure_rate):
    phase_sequence_index = 0
    flag = True
    while flag:
        phase = phase_sequence[phase_sequence_index % len(phase_sequence)]
        phase_loc = return_phase_loc(phase)
        all_zero_columns = np.all(Arrival_Table == 0, axis=0)
        if all_zero_columns[phase_loc].all():
            phase_sequence_index += 1
            continue
        else:
            flag = False
        phase_loc_1, phase_loc_2, phase_loc_3 = return_phase_loc_spec(phase)
        prev_stage_calculation = []  # 20240123 record previous traffic flow
        # first +1 because we need one more +1, the second +1 because we need to obtain the value
        for i in range(G_min_T[phase]+Yellow+Red+1, G_max_T+Yellow+Red+1+1): #first stage (phase), stage variable to Gmax
            s1c_temp = np.zeros((i, approaches+2))
            s1c_temp[G_min_T[phase]+Yellow+Red:i, -2] = np.arange(G_min_T[phase], i-5)
            s1c_temp[0, :approaches] = Arrival_Table[0, :approaches]
            s1c_temp[0, -1] = np.sum(s1c_temp[0, :approaches])
            for j in range(1, i-5):
                s1c_temp[j, :approaches] = s1c_temp[j-1,:approaches]+Arrival_Table[j, :approaches]
                s1c_temp[j, phase_loc_1] = np.maximum(s1c_temp[j, phase_loc_1] - departure_rate, 0)
                s1c_temp[j, phase_loc_2] = np.maximum(s1c_temp[j, phase_loc_2] - departure_rate*2, 0)
                s1c_temp[j, phase_loc_3] = np.maximum(s1c_temp[j, phase_loc_3] - departure_rate*3, 0)
                s1c_temp[j, -1] = s1c_temp[j-1, -1] + np.sum(s1c_temp[j, :approaches])
            for j in range(i-5, len(s1c_temp)):
                s1c_temp[j, :approaches] = s1c_temp[j - 1,:approaches] + Arrival_Table[j, :approaches]
                s1c_temp[j, -1] = s1c_temp[j-1, -1] + np.sum(s1c_temp[j, :approaches])
            prev_stage_calculation.append(s1c_temp)
    # Build decision table
    historical_decision_table = np.zeros((plan_horizon+1, 1))
    historical_decision_table[0:len(prev_stage_calculation[-1])] = prev_stage_calculation[-1][:, -2].reshape(-1, 1)
    # Build value function table
    historical_value_table = np.zeros((plan_horizon+1, 1))
    historical_value_table[0:len(
        prev_stage_calculation[-1])] = prev_stage_calculation[-1][:, -1].reshape(-1, 1)
    # Signal
    historical_signals = [phase]
    # stage>=1
    flag = True  # flag for stop
    # left_stage_num=5
    s_start=G_min_T[phase] + Yellow + Red #the start of stage variable of next phase is the sum of G_Min of current and next phases plus clearance
    phase_sequence_index += 1
    offset = G_min_T[phase]
    while flag:
        phase = phase_sequence[phase_sequence_index % len(phase_sequence)] #current phase
        phase_loc = return_phase_loc(phase)
        # check phase to skip
        all_zero_columns = np.all(Arrival_Table == 0, axis=0)
        if all_zero_columns[phase_loc].all():
            phase_sequence_index += 1
            continue
        
        s_start+=G_min_T[phase]+Yellow+Red 
        # Start
        historical_signals.append(phase)
        # build decision table
        decision_table = np.zeros((plan_horizon+1, 1))
        # Build value function table as a NumPy array
        value_table = np.zeros((plan_horizon+1, 1))

        # avoid the impact of the update of prev_stage_calculation
        prev_stage_calculation_previous = prev_stage_calculation.copy()
        # allow max green time -G_min_T[phase]
        for s in range(s_start, min(s_start+G_max_T-offset+1, 121)):
            #print(s)
            x, v, stage, replacement = signal_DP_sup_1(s, phase, prev_stage_calculation_previous, G_min_T, G_max_T, Arrival_Table, s_start, offset)
            decision_table[s] = x
            value_table[s] = v
            if len(stage) <= len(prev_stage_calculation[-1]):
                # G_min_T[phase_sequence[phase_sequence.index(phase)-1]]
                prev_stage_calculation[s-Yellow-Red-offset] = stage.copy()
            else:
                prev_stage_calculation.append(stage.copy())
        historical_decision_table = np.concatenate(
            (historical_decision_table, decision_table), axis=1)
        historical_value_table = np.concatenate(
            (historical_value_table, value_table), axis=1)

        # finish one stage
        if s == 120:
            flag = False

        phase_sequence_index += 1
    return historical_decision_table, historical_value_table, historical_signals


def optimal_policy_generation(result, plan_horizon):
    stage_index = len(result[2])-1
    time_index = plan_horizon
    time_all = []
    for i in range(len(result[2])):
        time_temp = result[0][int(time_index), int(stage_index)]
        time_all.append(time_temp)
        stage_index -= 1
        if time_temp == 0:
            time_index = max(time_index-time_temp, 0)
        else:
            time_index = max(time_index-time_temp-5, 0)
    time_all.reverse()
    signal_time = [[a, b*10] for a, b in zip(result[2], time_all)]
    flag = True
    while flag:
        if signal_time[0][0] != signal_time[1][0]:
            flag = False
        else:
            signal_time[1][1] += signal_time[0][1]
            signal_time.pop(0)
            if signal_time[0][1] >= 400:
                signal_time[0][1] = 400  # bug fixed 400ms not 40s
                flag = False
    new_list = []
    for i in range(len(signal_time)):
        if signal_time[i][1] != 0:
            signal_return = return_signal(signal_time[i][0])
            new_list.append([signal_return[0], signal_time[i][1]])
            new_list.append([signal_return[1], 40])
            new_list.append([signal_return[2], 10])
    return new_list

def update_CAV_flags(veh_ids: list[str]) -> None:

    for vid in veh_ids:
        if vid not in cav_flag:
            cav_flag[vid] = random.random() < PENETRATION_RATE   # True / False

def get_observed_vehicle_ids(veh_ids: list[str]) -> list[str]:

    if not veh_ids:
        return []

    positions = {v: traci.vehicle.getPosition(v) for v in veh_ids}

    dims = {}
    for v in veh_ids:
        try:
            dims[v] = (traci.vehicle.getWidth(v),
                       traci.vehicle.getLength(v),
                       traci.vehicle.getHeight(v))
        except Exception:
            dims[v] = (traci.vehicle.getWidth(v),
                       traci.vehicle.getLength(v),
                       1.5)

    # ---- loop over CAVs currently in the simulation ------------------------
    observed: set[str] = set()
    current_cavs       = [v for v in veh_ids if cav_flag.get(v, False)]

    for cav in current_cavs:
        cx, cy = positions[cav]

        # neighbours within PERCEPTION_RANGE (except the cav itself)
        neigh = [v for v in veh_ids if v != cav and
                 math.hypot(positions[v][0] - cx,
                            positions[v][1] - cy) <= PERCEPTION_RANGE]

        if not neigh:
            observed.add(cav)              # at least keep the cav itself
            continue

        def prob_visible(d: float) -> float:
            if   d < 10:             return 0.969
            elif d >=10 and d < 20:  return 0.931
            elif d >=20 and d < 30:  return 0.774
            elif d >=30 and d < 40:  return 0.566
            elif d >=40 and d < 50:  return 0.353
            elif d >=50 and d < 55:  return 0.207
            else:                    return 0.0
            
        visible_neigh = []
        
        # Random Drop
        for vid in neigh:
            d = math.hypot(positions[vid][0] - cx,
                           positions[vid][1] - cy)
            if random.random() < prob_visible(d):
                visible_neigh.append(vid)
        
        
        
        observed.update(visible_neigh)
        observed.add(cav)
    return list(observed)
  
# Start SUMO
if __name__=='__main__':

    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    
    
    # -----------------------  CAV-RELATED PARAMETERS  --------------------------
    PENETRATION_RATE  = 0.03    # 30 % of all vehicles become CAVs
    PERCEPTION_RANGE  = 54.0   # [m] radial sensing range of a CAV
    RANDOM_SEED       = 111     # reproducible sampling
    random.seed(RANDOM_SEED)
    
    # -----------------------  CAV-RELATED PARAMETERS  --------------------------
    BIN_SIZE         = 10.0
    EDGES            = np.arange(0, PERCEPTION_RANGE + BIN_SIZE, BIN_SIZE)
    VIS_COUNTS_GLOBAL   = np.zeros(len(EDGES) - 1, dtype=int)
    TOTAL_COUNTS_GLOBAL = np.zeros(len(EDGES) - 1, dtype=int)

    cav_flag: dict[str, bool] = {}   # keeps the CAV label for every vehicle ever seen
    
    # --------------------------  SUMO-RELATED PARAMETERS  -------------------------
    # Parameters setting
    traffic_light_id = '79'
    junction_id = "79"
    # Green Time
    G_min_T = {'012': 5, '34': 5, '567': 5, '8910': 5}#G_min_T = {'012': 10, '34': 10, '567': 10, '8910': 10}
    G_max_T = 40
    Yellow = 4
    Red = 1
    # Mapping phase from lanes
    depart_lane_list = ['173166881_0', '173166881_1', '173166881_2', # South Bound
                        'E0_0', 'E0_1', # North Bound
                        '174262747_0', '174262747_1', '174262747_2', # East Bound
                        '173896171_0', '173896171_1', '173896171_2', 
                        '655620659_0', '655620659_1', '655620659_2', '655620659_3'] # West Bound
    intersection_center = [238.96, 255.79]
    plan_horizon = 120
    approaches = 11
    departure_rate = 0.5
    phase_sequence = ['8910', '34', '012', '567']  # 20240123

    sumoCmd = ["sumo-gui", "-c", "./osm.sumocfg"]
    traci.start(sumoCmd)
    routes = {}
    flag = False
    signal_list = []
    Arrival_Table_list = []
    result_list = []
    marker = 1000
    for step in range(36000):  # stepwidth=0.1
        print(step)
        traci.simulationStep()  # Advance the simulation
        
        # -------------------  ▼▼  NEW CAV LOGIC ▼▼  ------------------------
        # 1) all vehicles currently in the simulation
        all_ids = traci.vehicle.getIDList()

        # 2) update the global cav_flag dict for *new* vehicles
        update_CAV_flags(all_ids)

        # 3) keep only vehicles seen by at least one CAV
        veh_id_list = get_observed_vehicle_ids(all_ids)
        
        # 4) Coloring
        observed_set = set(veh_id_list)
        for vid in all_ids:
            if cav_flag[vid]:
                # CAVs → red
                traci.vehicle.setColor(vid, (255, 0, 0, 255))
            elif vid in observed_set:
                # observed non‐CAVs → blue
                traci.vehicle.setColor(vid, (0, 0, 255, 255))
            else:
                # unobserved → light gray
                traci.vehicle.setColor(vid, (255, 255, 255, 255))
        
        # Warm up
        if step == 1000:
            flag = True
            # Warm up end
        if step % 10 == 0:
            #veh_id_list = traci.vehicle.getIDList()
            for veh_id in veh_id_list:
                if veh_id not in routes:
                    routes[veh_id] = []
                lane_id = traci.vehicle.getLaneID(veh_id)
                if lane_id in depart_lane_list or lane_id[:3] == ':79':
                    del routes[veh_id]
                else:
                    routes[veh_id].append(lane_id)
        # DP
        if flag and step == marker:
            # Arrival table
            # Creating a len(plan_horizon)xlen(approaches) DataFrame
            Arrival_Table = np.zeros((plan_horizon+1, approaches))
            # get vehicle id list
            for veh_id in routes:
                if veh_id in veh_id_list:
                    loc = return_ETA_cell(
                        veh_id, intersection_center, plan_horizon, routes)
                    if loc:
                        Arrival_Table[loc[0], loc[1]] += 1
            Arrival_Table_list.append(Arrival_Table)
            if np.all(Arrival_Table==0):
                signal=[[return_signal(phase_sequence[0])[0],G_min_T[phase_sequence[0]]*10],[return_signal(phase_sequence[0])[0]+1,40],[return_signal(phase_sequence[0])[0]+2,10]]
            else:
                result = signal_DP(Arrival_Table, G_min_T, G_max_T,
                                phase_sequence, plan_horizon, approaches, departure_rate)
                result_list.append(result)
                signal = optimal_policy_generation(result, plan_horizon)
            print(signal)
            for i in range(3):
                marker += signal[i][1]
            index=phase_sequence.index(return_phase_char(signal[0][0]))
            phase_sequence = phase_sequence[index+1:]+phase_sequence[:index+1]
            signal_list.append(signal)  # check

            time_list = [signal[0][1]+step]
            for i in range(1, len(signal)):
                time_list.append(signal[i][1]+time_list[i-1])

        if flag == True:
            for i in range(len(time_list)):
                if i == 0 and step < time_list[i]:
                    traci.trafficlight.setPhase(traffic_light_id, signal[i][0])
                elif step >= time_list[i-1] and step < time_list[i]:
                    traci.trafficlight.setPhase(
                        traffic_light_id, signal[i][0])  # works!
    traci.close()
