# save this as check_events.py in your project folder

import os
import mne

# Path to your dataset
data_path = r"C:\Users\shari\Videos\nubrafusex\data\BCICIV_2a_gdf"

# Loop through all .gdf files
for filename in os.listdir(data_path):
    if filename.endswith(".gdf") and not filename.startswith("~$"):
        filepath = os.path.join(data_path, filename)
        try:
            raw = mne.io.read_raw_gdf(filepath, preload=False, verbose='ERROR')
            events, event_id = mne.events_from_annotations(raw)
            print(f"\nFile: {filename}")
            print("Events found:", event_id)
        except Exception as e:
            print(f"\nFailed to load {filename}: {e}")
