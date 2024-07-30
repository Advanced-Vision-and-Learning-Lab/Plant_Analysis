from plantcv import plantcv as pcv
import cv2
import plantcv
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from skimage.filters import threshold_li

def get_plant_phenotypes(original_image, offset):

    # Set debug to the global parameter 
    pcv.params.debug = None
    gray_image = pcv.rgb2gray(rgb_img = original_image)
    binary_threshold = threshold_li(gray_image)
    binary_image = gray_image > binary_threshold
    binary_image = binary_image.astype(int)
    
    filled_binary_image = pcv.fill(bin_img = binary_image, size = 10)
    
    object_contours, object_hierarchies = pcv.find_objects(img = np.uint8(original_image), mask = filled_binary_image)
    rectangle_roi_contour, rectangle_roi_hierarchy= pcv.roi.rectangle(img = original_image, x = 95, y = 5, h = 500, w = 350)
    roi_object_contours, roi_object_hierarchies, roi_mask, roi_object_areas = pcv.roi_objects(img = original_image,
                                                                   roi_contour = rectangle_roi_contour, 
                                                                   roi_hierarchy = rectangle_roi_hierarchy, 
                                                                   object_contour = object_contours, 
                                                                   obj_hierarchy = object_hierarchies,
                                                                   roi_type = 'partial')
    
    composed_object, composed_mask = pcv.object_composition(img = original_image,
                                                            contours = roi_object_contours,
                                                            hierarchy = roi_object_hierarchies)
    
    analyze_object_outputs = pcv.analyze_object(img = original_image,
                                             obj = composed_object,
                                             mask = composed_mask)

    skeleton = pcv.morphology.skeletonize(composed_mask)

    segmented_image, segmented_objects = pcv.morphology.segment_skeleton(skel_img = skeleton, mask = composed_mask)

    secondary_objects, primary_objects = pcv.morphology.segment_sort(skel_img = skeleton,
                                                                     objects = segmented_objects,
                                                                     mask = composed_mask,
                                                                     first_stem=True)

    #labeled_image = pcv.morphology.segment_path_length(segmented_image, segmented_objects)

    num_branches = len(primary_objects)
    num_leaves = len(secondary_objects)

    plant_height_pixels = pcv.outputs.observations['default']['height']['value']
    plant_width_pixels = pcv.outputs.observations['default']['width']['value']
    plant_area_pixels = pcv.outputs.observations['default']['area']['value']
    plant_perimeter_pixels = pcv.outputs.observations['default']['perimeter']['value']
    plant_solidity = pcv.outputs.observations['default']['solidity']['value']
    
    image_width, image_height, image_channels = original_image.shape
    
    actual_greenhouse_height = plant_height_pixels * offset
    actual_greenhouse_width = plant_width_pixels * offset
    actual_greenhouse_area = plant_area_pixels * offset * offset
    actual_greenhouse_perimeter = plant_perimeter_pixels * offset

    # Create a dict with the plant phenotypes
    data = {'Image Height (pixels)': image_height,
              'Image Width (pixels)': image_width,
              'Pixel Size (cm)': offset,
              'Plant Height (pixels)': plant_height_pixels,
              'Plant Height (cm)': actual_greenhouse_height,
              'Plant Width (pixels)': plant_width_pixels,
              'Plant Width (cm)': actual_greenhouse_width,
              'Plant Area (pixels)': plant_area_pixels,
              'Plant Area (square cm)': actual_greenhouse_area,
              'Plant Perimeter (cm)': actual_greenhouse_perimeter,
              'Plant Perimeter (pixels)': plant_perimeter_pixels,
              'Plant Solidity': plant_solidity,
              'Number of Branches': num_branches,
              'Number of Leaves': num_leaves}
    
    return data
