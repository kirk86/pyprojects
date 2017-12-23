
import numpy as np
import glob2

import torch
import torch.utils.data as data
import torchbiomed.utils as utils
from glob import glob
import os
import os.path
import SimpleITK as sitk
import pandas as pd

import matplotlib.pyplot as plt
import functools
import multiprocessing

# In[10]:

def normalize_lung_CT(zmax, ymax, xmax, vox_spacing, src, dst, idx):
    mean_values = []
    var_values = []
    MIN_BOUND = -1000
    MAX_BOUND = 400
    Z_MAX, Y_MAX, X_MAX = zmax, ymax, xmax
    vox_spacing = vox_spacing
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    luna_subset_path = src
    luna_save_path = dst
    file_list=glob2.glob(luna_src + 'subset*/*.mhd')
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)

    for img_file in range(len(file_list)):
        itk_img = sitk.ReadImage(file_list[img_file])
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img, mu, var = utils.resample_volume(img_array, spacing_old, img_spacing, bounds=(MIN_BOUND, MAX_BOUND))
        utils.save_updated_image(img, itk_img, luna_save_path+os.path.basename(file_list[img_file]), img_spacing)
        mean_values.append(mu)
        var_values.append(var)
    dataset_mean = np.mean(mean_values)
    dataset_stddev = np.sqrt(np.mean(var_values))
    return (dataset_mean, dataset_stddev)

def normalize_lung_mask(zmax, ymax, xmax, vox_spacing, src, dst, idx):
    Z_MAX, Y_MAX, X_MAX = zmax, ymax, xmax
    vox_spacing = vox_spacing
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    luna_seg_lungs_path = src
    luna_seg_lungs_save_path = dst
    file_list=glob2.glob(luna_src + 'subset*/*.mhd')
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)
    for img_file in range(len(file_list)):
        itk_img = sitk.ReadImage(file_list[img_file])
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        spacing_old = (z_space, y_space, x_space)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img, _, _ = utils.resample_volume(img_array, spacing_old, img_spacing)
        img[img < 1] = 0
        utils.save_updated_image(img, itk_img,
                                 os.path.join(luna_seg_lungs_save_path, os.path.basename(file_list[img_file])),
                                 img_spacing)


# In[11]:

def build_nodule_offset_tables(file_list, save_path):
    offset_dict = {}
    origin_dict = {}
    for img_file in file_list:
        series = os.path.basename(img_file)[:-4]
        itk_img = sitk.ReadImage(img_file)
        (x_space, y_space, z_space) = itk_img.GetSpacing()
        img_spacing = (z_space, y_space, x_space)
        x_size, y_size, z_size = itk_img.GetSize()
        img_size = (z_size, y_size, x_size)
        resize_factor = np.array(img_spacing) / [vox_spacing, vox_spacing, vox_spacing]
        (new_z, new_y, new_x) = np.round(img_size * resize_factor)
        z_off = int((new_z - Z_MAX)/2)
        y_off = int((new_y - Y_MAX)/2)
        x_off = int((new_x - X_MAX)/2)
        offset_dict[series] = np.array((z_off, y_off, x_off))
        x_orig, y_orig, z_orig = itk_img.GetOrigin()
        origin_dict[series] = np.array((z_orig, y_orig, x_orig))
    utils.npz_save(os.path.join(save_path, "origin_table"), origin_dict)
    utils.npz_save(os.path.join(save_path, "offset_table"), offset_dict)
    
def normalize_nodule_mask(annots, zmax, ymax, xmax, vox_spacing, tables, src, dst, idx):

    def get_boundaries(origin, offsets, params):
        diam, center = params
        diam3 = np.array((diam, diam, diam))
        diamu = diam + vox_spacing
        diam3u = np.array((diamu, diamu, diamu))
        v_center = np.rint((center - origin)/vox_spacing)
        v_lower = np.rint((center - diam3 - origin)/vox_spacing)
        v_upper = np.rint((center + diam3u - origin)/vox_spacing)
        v_center -= offsets
        v_lower -= offsets
        v_upper -= offsets
        x_list.append(v_upper[2])
        y_list.append(v_upper[1])
        z_list.append(v_upper[0])
        x_list.append(v_lower[2])
        y_list.append(v_lower[1])
        z_list.append(v_lower[0])
        return (v_lower, v_center, v_upper)

    def l2_norm(pointA, pointB):
        point = pointA - pointB
        return np.sqrt(np.dot(point, point))

    def get_filename(case):
        for f in file_list:
            if case in f:
                return(f)

    def update_mask(mask, CT, bounds):
        v_lower, v_center, v_upper = bounds
        z_min, y_min, x_min = v_lower
        z_max, y_max, x_max = v_upper
        pixel_count = 0
        min_ct = np.min(CT)
        radius = np.rint((z_max - z_min + vox_spacing)/2)
        ct_thresh = min_ct + 1
        bit_count = 0
        #print(v_center)
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if l2_norm(np.array((z, y, x)), v_center) > radius:
                        break
                    if CT[z][y][x] > ct_thresh:
                        mask[z][y][x] = 4
                        pixel_count += 1
                        bit_count += 1
        assert bit_count != 0
    
    pixel_count = 0
    mask_count = 0
    annotations = annots
    Z_MAX, Y_MAX, X_MAX = zmax, ymax, xmax
    shape_max = (Z_MAX, Y_MAX, X_MAX)
    vox_spacing = vox_spacing
    tables_path = tables
    utils.init_dims3D(Z_MAX, Y_MAX, X_MAX, vox_spacing)
    x_list, y_list, z_list = [], [], []
    luna_normal_path = src
    luna_mask_path = dst

    #file_list = glob2.glob(src + 'subset*/*.mhd')
    file_list = glob(src + '/*.mhd')
    print(file_list)
    df_node = pd.read_csv(annotations)
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()
    img_spacing = (vox_spacing, vox_spacing, vox_spacing)
    count = 0
    
    origin_dict = utils.npz_load(os.path.join(tables_path, "origin_table"))
    offset_dict = utils.npz_load(os.path.join(tables_path, "offset_table"))
    
   
    for img_file in range(len(file_list)):
        mask_count = 0
        mini_df = df_node[df_node["file"]==file_list[img_file]] #get all nodules associate with file
        if len(mini_df) == 0:
            continue
        mask = np.zeros(shape_max, dtype=np.int16)
        series = os.path.basename(file_list[img_file])[0:-4]
        origin = origin_dict[series]
        if origin[1] > 0 and origin[2] > 0:
            origin[1] = -origin[1]
            origin[2] = -origin[2]
        offsets = offset_dict[series]
        itk_img = sitk.ReadImage(file_list[img_file])
        img_array = sitk.GetArrayFromImage(itk_img)
        print('Image shape:', img_array.shape)
        for i in range(len(mini_df)):
            node_x = mini_df["coordX"].values[i]
            node_y = mini_df["coordY"].values[i]
            node_z = mini_df["coordZ"].values[i]
            diam = mini_df["diameter_mm"].values[i]
            params = (diam, np.array((node_z, node_y, node_x)))
            bounds = get_boundaries(origin, offsets, params)
            _, v_center, _ = bounds
            if np.min(v_center) < 0:
                print("origin: {} offsets: {}\n params: {} v_center: {}".format(
                    origin, offsets, params, v_center))
                continue
            bounds = np.array(bounds).astype(np.int16)
            update_mask(mask, img_array, bounds)
            mask_count += 1
        assert mask_count != 0
        itk_mask_img = sitk.GetImageFromArray(mask, isVector=False)
        itk_mask_img.SetSpacing(img_spacing)
        itk_mask_img.SetOrigin(origin)
        sitk.WriteImage(itk_mask_img, luna_mask_path+'/'+os.path.basename(file_list[img_file]))
    return

def run(tables = False):
    
    if tables:
        tables_save = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/LUNA_pytorch/tables/'
        build_nodule_offset_tables(pats, tables_save)

        tables_save = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/LUNA_pytorch/tables_ct/'
        build_nodule_offset_tables(pats1, tables_save)

    normalize_nodule_mask(zmax = 160, ymax = 128, xmax = 160, vox_spacing = 2.5,
         annots = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/evaluationScript/annotations/annotations.csv',
        dst = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_nodule_mask/',
        tables = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/LUNA_pytorch/tables/',
        src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_ct/')
    return


MIN_BOUND = -1000
MAX_BOUND = 400

image_dict = {}
label_dict = {}
test_split = []
train_split = []

Z_MAX = 160
Y_MAX = 128
X_MAX = 160
vox_spacing = 2.5

luna_src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/'
ct1 = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_ct/'

pats = glob2.glob(luna_src + 'subset*/*.mhd')
pats1 = glob(ct1 + '*mhd')


zmax = 160
ymax = 128
xmax = 160
annots = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/evaluationScript/annotations/annotations.csv'
dst_lung = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_lung_mask/'
dst_nodule = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_nodule_mask/'
tables = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/LUNA_pytorch/tables/'
src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_ct/'

count = len(pats)
assert count > 0, 'Could not find %s' % search_path
print('Converting %d scans...' % count)

cpu_count = multiprocessing.cpu_count()
batch_size = (count - 1) // cpu_count + 1
processes = (count - 1) // batch_size + 1

class runner:

    def run_nodules():
        func = functools.partial(normalize_nodule_mask, annots, zmax, ymax, xmax, vox_spacing, tables, src, dst_nodule)
        pool = multiprocessing.Pool(processes=processes)
        ret_list = pool.map(func, range(processes))
        pool.close()
        return

    def run_ct():
        func = functools.partial(normalize_lung_CT, zmax, ymax, xmax, vox_spacing, src, dst_ct)
        pool = multiprocessing.Pool(processes=processes)
        ret_list = pool.map(func, range(processes))
        pool.close()
        return

    def run_lungs():
        func = functools.partial(normalize_lung_mask, zmax, ymax, xmax, vox_spacing, src, dst_lung)
        pool = multiprocessing.Pool(processes=processes)
        ret_list = pool.map(func, range(processes))
        pool.close()
        return
    
#runner.run_nodules()
#runner.run_lungs()
normalize_nodule_mask(zmax = 160, ymax = 128, xmax = 160, vox_spacing = 2.5,
         annots = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/evaluationScript/annotations/annotations.csv',
        dst = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_nodule_mask/',
        tables = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/LUNA_pytorch/tables/',
        src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/biomed_ct', idx = 0)
