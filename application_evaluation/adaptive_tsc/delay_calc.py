import xml.etree.ElementTree as ET

# 1) load your tripinfo output
#tree = ET.parse('./sumo/MIDAR_noRH_seed666.xml')
#tree = ET.parse('./sumo/MIDAR_seed666.xml')
tree = ET.parse('./sumo/RandomDrop_seed666.xml')
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