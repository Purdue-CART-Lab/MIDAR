#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 18:18:02 2025

@author: idiot
"""
import xml.etree.ElementTree as ET

# 1) load your tripinfo output
tree = ET.parse('./adaptive_tsc/sumo/MIDAR_noRH_seed18.xml')
#tree = ET.parse('./adaptive_tsc/sumo/MIDAR_seed18.xml')
root = tree.getroot()

delays = []
for trip in root.findall('tripinfo'):
    depart = float(trip.get('depart'))
    if depart > 100:                    # skip warm-up
        delays.append(float(trip.get('timeLoss')))

if delays:
    avg_delay = sum(delays) / len(delays)
    print(f"Avg. delay (post-warm-up): {avg_delay:.2f} s over {len(delays)} vehicles")
else:
    print("No trips after warm-up.")