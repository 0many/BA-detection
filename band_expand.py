import os
import numpy as np
import cv2
from osgeo import gdal


img_dir = r'D:\fire\change_fire_db\test_2020\a\4_S\input'
save_img_dir = r'D:\fire\change_fire_db\test_2020\a\5\ndvi\input'


def createDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def NDVI(arr):
    red = arr[0, :, :].astype(float)
    nir = arr[1, :, :].astype(float)
    NDVI = (nir - red) / (nir + red)
    return NDVI

def NBR(arr):
    nir = arr[1, :, :].astype(float)
    swir = arr[3, :, :].astype(float)
    NBR = (nir - swir) / (nir + swir)
    return NBR

def BAI(arr):
    red = arr[0, :, :]
    nir = arr[1, :, :]
    BAI = 1 / (((0.1 - red) ** 2) + ((0.06 - nir) ** 2))
    return BAI

def FBI(arr):
    red = arr[0, :, :].astype(float)
    nir = arr[1, :, :].astype(float)
    green = arr[2, :, :].astype(float)
    FBI = ((red / green) - nir) / ((red / green) + nir)
    return FBI


def Histogram_stretching(arr):
    arr_2 = np.percentile(arr[~np.isnan(arr)], 2)
    arr_98 = np.percentile(arr[~np.isnan(arr)], 98)
    arr[arr < arr_2] = arr_2
    arr[arr > arr_98] = arr_98
    return arr

createDir(save_img_dir)

for file_name in os.listdir(img_dir):

    post_img_path = os.path.join(img_dir, file_name)
    pre_img_path = post_img_path.replace('post', 'pre')

    ##img 열기 및 array 변환
    post_img = gdal.Open(post_img_path)
    post_arr = post_img.ReadAsArray()

    pre_img = gdal.Open(pre_img_path)
    pre_arr = pre_img.ReadAsArray()

    ##index 계산
    post_ndvi = NDVI(post_arr)
    pre_ndvi = NDVI(pre_arr)
    post_nbr = NBR(post_arr)
    pre_nbr = NBR(pre_arr)
    bai = BAI(post_arr)
    fbi = FBI(post_arr)
    dndvi = pre_ndvi - post_ndvi
    dnbr = pre_nbr - post_nbr


    ## 0-255값으로 스케일링
    scaled_ndvi = cv2.normalize(post_ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    scaled_nbr = cv2.normalize(post_nbr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    scaled_bai = cv2.normalize(bai, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    scaled_fbi = cv2.normalize(fbi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    scaled_dndvi = cv2.normalize(dndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    scaled_dnbr = cv2.normalize(dnbr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    ##Histogram stretching
    ndvi = Histogram_stretching(scaled_ndvi)
    nbr = Histogram_stretching(scaled_nbr)
    bai = Histogram_stretching(scaled_bai)
    fbi = Histogram_stretching(scaled_fbi)
    dndvi = Histogram_stretching(scaled_dndvi)
    dnbr = Histogram_stretching(scaled_dnbr)

    ##stack
    new_ras_arr = np.stack([post_arr[0, :, :], post_arr[1, :, :],post_arr[2, :, :],post_arr[3, :, :], ndvi])

    ##드라이버 설정
    driver = gdal.GetDriverByName('GTiff')

    ##new img 생성
    out_ras = driver.Create(os.path.join(save_img_dir, file_name), post_img.RasterXSize, post_img.RasterYSize, new_ras_arr.shape[0], gdal.GDT_Byte)

    ##밴드 데이터 및 img 저장
    for i in range(new_ras_arr.shape[0]):
        out_ras_band = out_ras.GetRasterBand(i + 1)
        out_ras_band.WriteArray(new_ras_arr[i])







