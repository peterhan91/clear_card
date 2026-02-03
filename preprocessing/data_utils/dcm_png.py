import os
import cv2
import pydicom
from glob import glob
from tqdm import tqdm
# Convert DICOM files to PNG format with histogram equalization
# and pixel normalization.

dcm_file_paths = glob('/home/than/Datasets/CXR_VQA/OD/physionet.org/files/vindr-cxr/1.0.0/train/*.dicom')
dcm_file_paths.sort()
output_dir = '/home/than/Datasets/CXR_VQA/OD/physionet.org/files/vindr-cxr/1.0.0/train_png/'
if not os.path.exists(output_dir):
   os.makedirs(output_dir)

for dcm_file_path in tqdm(dcm_file_paths):
   dcm_file = pydicom.dcmread(dcm_file_path)
   raw_image = dcm_file.pixel_array

   # Normalize pixels to be in [0, 255].
   rescaled_image = cv2.convertScaleAbs(dcm_file.pixel_array,
                                       alpha=(255.0/dcm_file.pixel_array.max()))

   # Correct image inversion.
   if dcm_file.PhotometricInterpretation == "MONOCHROME1":
      rescaled_image = cv2.bitwise_not(rescaled_image)
   # Perform histogram equalization.
   adjusted_image = cv2.equalizeHist(rescaled_image)

   output_file_path = os.path.join(output_dir, os.path.basename(dcm_file_path).replace('.dicom', '.png'))
   _ = cv2.imwrite(output_file_path, adjusted_image)