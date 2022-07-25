import os
import SimpleITK as sitk

def dicom_to_nifti(in_path,out_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(in_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    sitk.WriteImage(image,out_path)

path = '/home/claire/data'

l = ['COVID','Normal','PNA']

#suffix = set()

#dicom_to_nifti(os.path.join(path,'COVID','5M9RRKD5'),'leo.nii.gz')
#dicom_to_nifti(os.path.join(path,'PNA','5M9TIPWC'),os.path.join(path,'PNA_nifti', '5M9TIPWC.nii.gz'))

from tqdm import tqdm

f = open("error.txt", "a")

for i in l:
    files = os.listdir(os.path.join(path,i))
    os.makedirs(os.path.join(path,i+'_nifti'),exist_ok=True)
    for j in tqdm(range(len(files))):
        try:
            dicom_to_nifti(os.path.join(path,i,files[j]),os.path.join(path,i+'_nifti',files[j]+'.nii.gz'))
        except:
            f.write(f"Error in {files[j]} \n")

f.close()
