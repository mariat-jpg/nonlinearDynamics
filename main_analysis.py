from src.load_data import load_fmri, preprocess_fmri, load_atlas
from src.extract_bold import extract_bold_timeseries
import numpy as np

fmri_path = "dataset/sub-17017/func/sub-17017_task-rest_bold.nii.gz"

fmri_img = load_fmri(fmri_path)
fmri_clean = preprocess_fmri(fmri_img)

atlas_maps, atlas_labels = load_atlas()

bold_ts = extract_bold_timeseries(fmri_clean, atlas_maps)

np.save("outputs/bold_timeseries.npy", bold_ts)

print(bold_ts.shape) 