import numpy as np
import pandas as pd 
import pydicom
import cv2
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
import random
from PIL import Image
import glob
# Some constants 
INPUT_FOLDER = '/home/claire/data/dicom/COVID/'
OUTPUT_FOLDER = '/home/claire/data/processedoverlay2/COVID/'

# INPUT_FOLDER = '/home/claire/data/dicom/PNA/'
# OUTPUT_FOLDER = '/home/claire/data/processedoverlay2/PNA/'

# INPUT_FOLDER = '/home/claire/data/dicom/Normal/'
# OUTPUT_FOLDER = '/home/claire/data/connected-max/Normal/'

patients = os.listdir(INPUT_FOLDER)
print(len(patients))

# Load the scans in given folder path
def load_scan(path):
    series = ""
    # for s in os.listdir(path):
    #     if (not os.path.isdir(path)):
    #         series = pydicom.filereader.dcmread(path + '/' + s)
    #         print(series)
    #series = [pydicom.filereader.dcmread(path + '/' + s) for s in os.listdir(path) if (not os.path.isdir(path))]
    series = [pydicom.filereader.dcmread(path + '/' + s, force=True) for s in os.listdir(path) if (not s=='.DS_Store')]
    slices = []
    #print(os.listdir(path))
    # print(len(series))
    for i in range(len(series)):
        try:
            if(series[i].ImagePositionPatient[2]):
                slices.append(series[i])
        except:
            continue
    # print(slices)
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

#return scaled image, taken care of pixel values 
def get_pixels_hu(slices):
    # for s in slices:
    #     print(s.convert_pixel_data('gdcm'))
    
    # return
    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#max bound 400 bc it corresponds to lung tissues; above 400 bones will also appear
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
   
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def get_slices(image):
    slices = []
    pixmax = []
    for i in range(image.shape[2]):
        pixmax.append(np.sum(image.get_fdata()[:,:,i]))
        if np.sum(image.get_fdata()[:,:,i]) >= 1000:
            slices.append(i)
    #print(pixmax)
    #print(max(pixmax))
    return slices

n_slices = 10

def change_color(picture, width, height, num):
    # Process every pixel
    mask = np.zeros((width,height,3), dtype="uint8")
    for x in range(width):
        for y in range(height):
            if num == 1:
                if picture[x][y] == 1.0:
                    mask[y][x] = [255,0,0]
                elif picture[x][y] == 0.0:
                    mask[y][x] = [255,255,255]
                else:
                    print(picture[x][y])
            else:
                mask[x,y] = [picture[x][y],picture[x,y],picture[x,y]]
    return mask

def get_overlapped_img(image, mask):
    # Import orginal img
    width, height = mask.shape
    # Import and convert the mask from binary to RGB
    #mask = mask.convert('RGB')

    # Convert the white color (for blobs) to magenta
    mask_colored = change_color(mask, width, height, 1)
    image = change_color(image, width, height, 2)

    # Image.fromarray(image).save('image.png')
    #Image.fromarray(mask_colored).save(os.path.join(OUTPUT_FOLDER + patients[i] + '_mask.png'))

    return cv2.addWeighted(np.array(image),0.8,np.array(mask_colored),0.2,0)


COVID_SEG = '/home/claire/data/segmented2/COVID/'
PNA_SEG = '/home/claire/data/segmented2/PNA/'

for i in range(len(patients)):
    # dicom original
    if os.path.exists(OUTPUT_FOLDER+patients[i]): continue
    if patients[i]=='.DS_Store': continue
    first_patient = load_scan(INPUT_FOLDER + patients[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    first_patient_pixels = normalize(first_patient_pixels)

    #segmented nifti
    # first_patient = nib.load(INPUT_FOLDER + patients[i])
    # first_patient_pixels = get_pixels_hu(first_patient.dataobj)
    # first_patient_pixels = normalize(first_patient_pixels)

    # for normal slices, generate n_slices random numbers from center +5 -5
    # center = first_patient_pixels.shape[0] / 2
    # print(center)
    # k = int((first_patient_pixels.shape[0] * 0.4) / 2)
    # slicelist = range(int(center - k), int(center + k))
    
    # for COVID and PNA, slices where lesions are present, where voxel count > 10
    # go over each slice of 3d scan and count the lesion pixels
    image_folder = os.path.abspath(COVID_SEG)
    image = os.path.join(image_folder, patients[i], patients[i] + "_seg.nii.gz")
    img = nib.load(image)
    slicelist = get_slices(img)
    
    #print(len(slicelist))
    
    # original slice processing
    # for slice in slicelist:
    #     data = np.array(first_patient_pixels[slice])
    #     #Rescale to 0-255 and convert to uint8
    #     rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    #     im = Image.fromarray(rescaled)
    #     im.save(os.path.join(OUTPUT_FOLDER + patients[i] + '_slice_'+str(slice)+'.png'))
    
    # take information from neighboring slices
    # for idx in range(len(slicelist)):
    #     data = np.array(first_patient_pixels[slicelist[idx]])
    #     if ((idx - 2 >= 0) and (idx + 2 < len(slicelist))):
    #         datam2 = np.array(first_patient_pixels[slicelist[idx - 2]])
    #         datam1 = np.array(first_patient_pixels[slicelist[idx - 1]])
    #         datap1 = np.array(first_patient_pixels[slicelist[idx + 1]])
    #         datap2 = np.array(first_patient_pixels[slicelist[idx + 2]])
    #         data = np.maximum(np.maximum(np.maximum(np.maximum(datam2, datam1), data), datap1), datap2)
    #     elif ((idx - 1 >= 0) and (idx + 1 < len(slicelist))):
    #         datam1 = np.array(first_patient_pixels[slicelist[idx - 1]])
    #         datap1 = np.array(first_patient_pixels[slicelist[idx + 1]])
    #         data = np.maximum(np.maximum(datam1, datap1), data)
    #     elif ((idx + 1) < len(slicelist)):
    #         datap1 = np.array(first_patient_pixels[slicelist[idx + 1]])
    #         data = np.maximum(datap1, data)
    #     else:
    #         datam1 = np.array(first_patient_pixels[slicelist[idx - 1]])
    #         data = np.maximum(datam1, data)

    #     #Rescale to 0-255 and convert to uint8
    #     rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    #     im = Image.fromarray(rescaled)
    #     #nib.save(im, os.path.join(OUTPUT_FOLDER + patients[i], patients[i] + '_slice_'+str(slice)+'.nii.gz'))
    #     im.save(OUTPUT_FOLDER + patients[i] + '_slice_'+str(slicelist[idx])+'.png')
    
    # overlay slice processing
    for slice in slicelist:
        mask = img.get_fdata()[:,:,slice]
        data = np.array(first_patient_pixels[slice])
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        overlay = get_overlapped_img(rescaled, mask)
        Image.fromarray(overlay).save(os.path.join(OUTPUT_FOLDER + patients[i] + '_slice_'+str(slice)+'.png'))
        # im = Image.fromarray(overlay)
        # im.save(os.path.join(OUTPUT_FOLDER + patients[i] + '_slice_'+str(slice)+'.png'))
