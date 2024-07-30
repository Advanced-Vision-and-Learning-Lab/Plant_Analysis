import os
import cv2
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pdb
from Connect_Components_Preprocessing import CCA_Preprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Image_Stitching import *
from Plant_Phenotypes import *
from Image_Segmentation import *
from skimage.feature import local_binary_pattern,hog
from skimage import exposure
from time import time
import pickle
import json
import shutil
from plantcv import plantcv as pcv
import plantcv
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaml

file = open('pipeline_config.yaml', 'r')
pipeline_config = yaml.safe_load(file)
file.close()

class Plant_Analysis:
    
    def __init__(self, session):

        self.batch_size = pipeline_config['pipeline_batch_size']
        self.service_type = 0
        self.input_folder_path = None
        self.output_folder_path = None
        self.show_raw_images = False
        self.show_color_images = False
        self.plant_paths = {}
        self.plant_stats = {}
        self.interm_result_folder = 'Interm_Results_'+str(session)
        self.segmentation_model_weights_path = pipeline_config['segmentation_model_weights_path']
        self.segmentation_model = None
        self.variable_k = pipeline_config['cca_variable_k']
        self.raw_channel_names = ['Red (660 nm)', 'Green (580 nm)', 'Red Edge (730 nm)', 'NIR (820 nm)']
        self.device = pipeline_config['device']
        self.LBP_radius = pipeline_config['LBP_radius']
        self.LBP_n_points = 8*self.LBP_radius
        self.offset = pipeline_config['offset']
        self.analysis_items = ['stitched_image', 'cca_image', 'segmented_image', 'tips', 'branches', 'tips_and_branches', 'sift_features', 'lbp_features', 'hog_features', 'ndvi_image']
        self.statistics_items = ['Height', 'Width', 'Area', 'Perimeter', 'Solidity', 'Number of Branches', 'Number of Leaves', 'NDVI (Maximum)', 'NDVI (Minimum)', 'NDVI (Average)', 'NDVI (Positive Average)']
        self.statistics_units = [' cm', ' cm', ' square cm', ' cm', '', '', '', '', '', '', '']
    
    def update_service_type(self,service):
        
        self.service_type = service

    def check_for_ipynb(self, input_list):

        if '.ipynb_checkpoints' in input_list:
            input_list.remove('.ipynb_checkpoints')

        return input_list
    
    def parse_folders(self):

        print('Debug: parsing folders')
        if self.service_type == 0:
            folder_path = self.input_folder_path
            plant_folders = sorted(os.listdir(folder_path))
            for plant_folder in plant_folders:
                self.plant_paths[plant_folder] = {}
                self.plant_stats[plant_folder] = {}
                self.plant_paths[plant_folder]['raw_images'] = []
                plant_folder_path = os.path.join(folder_path,plant_folder)
                image_names = self.check_for_ipynb(sorted(os.listdir(plant_folder_path)))
                for image_name in image_names:
                    image_path = os.path.join(plant_folder_path,image_name)
                    self.plant_paths[plant_folder]['raw_images'].append(image_path)
                    
        if self.service_type == 1:
            plant_folder_path = self.input_folder_path
            plant_name = plant_folder_path.split('/')[-1]
            self.plant_paths[plant_name] = {}
            self.plant_stats[plant_name] = {}
            self.plant_paths[plant_name]['raw_images'] = []
            # self.plants[plant_folder]['raw_images'] = []
            image_names = self.check_for_ipynb(sorted(os.listdir(plant_folder_path)))
            for image_name in image_names:
                image_path = os.path.join(plant_folder_path,image_name)
                self.plant_paths[plant_name]['raw_images'].append(image_path)
                
    def update_input_path(self,input_path):
        
        self.input_folder_path = input_path
        self.parse_folders()
        
    def update_check_RI_option(self, check_RI):
        
        self.show_raw_images = check_RI

    def update_check_CI_option(self, check_CI):
        
        self.show_color_images = check_CI
    
    def load_segmentation_model(self):
        
        self.segmentation_model = load_yolo_model(self.segmentation_model_weights_path)

    def get_plant_names(self):

        return sorted(list(self.plant_paths.keys()))

    def get_raw_images(self, plant):

        return [(Image.open(image_path), image_path.split('/')[-1].split('.')[0]) for image_path in self.plant_paths[plant]['raw_images']]
        
    def get_color_images(self, plant):

        with open(self.plant_paths[plant]['color_images_pickle'], 'rb') as handle:
            color_images = pickle.load(handle)['color_images']
        return [(image.astype(np.uint8), image_name) for image,image_name in color_images]

    def get_segmented_image(self, plant):

        return cv2.imread(self.plant_paths[plant]['segmented_image'])

    def get_plant_analysis_images(self, plant):

        with open(self.plant_paths[plant]['plant_analysis_pickle'], 'rb') as handle:
            plant_analysis_dict = pickle.load(handle)
        return [(image.astype(np.uint8), image_name) for image,image_name in [plant_analysis_dict[item] for item in self.analysis_items]]

    def get_plant_height(self, plant):

        return str(round(self.plant_stats[plant]['Height'],2))+' cm'
    
    def get_plant_statistics_df_plantwise(self, plant):
        
        return pd.DataFrame({'Phenotypic trait': self.statistics_items, 'Value': [str(round(self.plant_stats[plant][self.statistics_items[index]],2))+self.statistics_units[index] for index in range(len(self.statistics_items))]})
        
    def tile(self, image, d=2):
        
        w, h = image.size
        grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
        boxes = []
        
        for i, j in grid:
            box = (j, i, j+d, i+d)
            boxes.append(box)

        return boxes

    def make_batches(self):

        self.batches = []
        plant_names = self.get_plant_names()
        num_plants = len(plant_names)
        num_batches = num_plants//self.batch_size
        
        for iter in range(num_batches+1):

            begin = self.batch_size*iter
            end = min(num_plants,self.batch_size*(iter+1))
            if end > begin:
                self.batches.append(plant_names[begin:end])
    
    def do_plant_analysis(self):

        self.make_batches()
        
        for batch in self.batches:
            
            self.plants = {}
            self.load_raw_images(batch)
            self.get_ndvi_image_indices(batch)
            self.make_color_images(batch)
            self.stitch_color_images(batch)
            self.calculate_connected_components(batch)
            self.run_segmentation(batch)
            self.calculate_plant_phenotypes(batch)
            self.calculate_tips_and_branches(batch)
            self.calculate_sift_features(batch)
            self.calculate_LBP_features(batch)
            self.calculate_HOG_features(batch)
            self.calculate_ndvi(batch)
            self.save_interm_result(batch)
            del self.plants

    def save_interm_result(self, batch):

        result_folder = self.interm_result_folder
        self.make_dir(result_folder)
        
        for plant_name in batch:

            plant_folder_path = os.path.join(result_folder,plant_name)
            self.make_dir(plant_folder_path)
            color_images_output_file_path = os.path.join(plant_folder_path,'color_images.pickle')
            self.plant_paths[plant_name]['color_images_pickle'] = color_images_output_file_path
            
            with open(color_images_output_file_path, 'wb') as handle:
                pickle.dump({'color_images': self.plants[plant_name]['color_images']}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
            del self.plants[plant_name]['raw_images']
            del self.plants[plant_name]['color_images']
            del self.plants[plant_name]['ndvi_image_index']
            del self.plants[plant_name]['ndvi_inputs']
            
            plant_analysis_output_file_path = os.path.join(plant_folder_path,'plant_analysis_images.pickle')
            self.plant_paths[plant_name]['plant_analysis_pickle'] = plant_analysis_output_file_path

            segmented_image_path = os.path.join(plant_folder_path,'segmented_image.jpg')
            cv2.imwrite(segmented_image_path,self.plants[plant_name]['segmented_image'][0])
            self.plant_paths[plant_name]['segmented_image'] = segmented_image_path

            with open(plant_analysis_output_file_path, 'wb') as handle:
                pickle.dump(self.plants[plant_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_raw_images(self, batch):

        for plant_name in batch:

            self.plants[plant_name] = {}
            self.plants[plant_name]['raw_images'] = []

            for image_path in self.plant_paths[plant_name]['raw_images']:

                image_name = image_path.split('/')[-1].split('.')[0]
                self.plants[plant_name]['raw_images'].append((Image.open(image_path), image_name))

    def get_ndvi_image_indices(self, batch):

        for plant_name in batch:

            num_images = len(self.plants[plant_name]['raw_images'])
            
            if num_images%2 == 0:
                index = (num_images//2) - 1
            else:
                index = (num_images//2)

            self.plants[plant_name]['ndvi_image_index'] = index
    
    def make_color_images(self, batch):
        
        for plant_name in batch:
            
            self.plants[plant_name]['color_images'] = []
            image_index = 0
            
            for raw_image, image_name in self.plants[plant_name]['raw_images']:
                
                size = raw_image.size[0] // 2
                slices = self.tile(raw_image, d = size)
                index = 0                
                image_stack = np.zeros((size, size, len(slices)))
                
                for box in slices:
                    
                    image_stack[:, :, index] = np.array(raw_image.crop(box))
                    index += 1
                    
                red = np.expand_dims(image_stack[:, :, 1], axis=-1)
                green = np.expand_dims(image_stack[:, :, 0], axis=-1)
                red_edge = np.expand_dims(image_stack[:, :, 2], axis=-1)
                NIR = np.expand_dims(image_stack[:, :, -1], axis=-1)
                
                composite_image = np.concatenate((green, red_edge, red), axis=-1) * 255
                normalized_image = ((composite_image - composite_image.min())*255 / (composite_image.max() - composite_image.min())).astype(np.uint8)
                
                self.plants[plant_name]['color_images'].append((normalized_image, image_name))

                if self.plants[plant_name]['ndvi_image_index'] == image_index:

                    self.plants[plant_name]['ndvi_inputs'] = {'red': red, 'NIR': NIR, 'color': normalized_image}

                image_index += 1

    def calculate_ndvi(self, batch):

        pcv.params.debug = None
        ndvi_min = -1.0
        ndvi_max = 1.0
        epsilon = pipeline_config['ndvi_epsilon']

        input_images = [self.plants[plant_name]['ndvi_inputs']['color'] for plant_name in batch]
        results = self.segmentation_model.predict(input_images, conf = pipeline_config['segmentation_confidence'], device = self.device)

        for result_index in range(len(results)):
            
            result = results[result_index]
            
            if result:

                plant_name = batch[result_index]
                if result.masks.data.shape[0] > 4:
                    result.masks.data = result.masks.data[:4]
                mask = preprocess_mask(result.masks.data)
                binary_mask_np = generate_binary_mask(mask)
                segmented_color_image = overlay_mask_on_image(binary_mask_np, input_images[result_index])
                original_image = segmented_color_image
                image = segmented_color_image
                gray_image = pcv.rgb2gray(rgb_img = image)
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
                
                masked_color_image = pcv.apply_mask(img = original_image, mask = composed_mask, mask_color = 'black')
                red = pcv.apply_mask(img = self.plants[plant_name]['ndvi_inputs']['red'], mask = composed_mask, mask_color = 'black')
                NIR = pcv.apply_mask(img = self.plants[plant_name]['ndvi_inputs']['NIR'], mask = composed_mask, mask_color = 'black')
                ndvi_image = (NIR-red)/(NIR+red+epsilon)
                #ndvi_image_normalized = ((ndvi_image - ndvi_min)*255 / (ndvi_max - ndvi_min)).astype(np.uint8)
                ndvi_image = pcv.apply_mask(img = ndvi_image, mask = composed_mask, mask_color = 'black')
                max_ndvi = ndvi_image.max()
                min_ndvi = ndvi_image.min()
                avg_ndvi = np.average(ndvi_image[ndvi_image != 0])
                pos_avg_ndvi = np.average(ndvi_image[ndvi_image > 0])
                #H,W,C = masked_ndvi_image.shape
                #masked_ndvi_image = np.reshape(masked_ndvi_image, (H,W))
                
                fig, ax = plt.subplots()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(ndvi_image, vmin = -1, vmax = 1, cmap = mpl.colormaps['RdYlGn'])
                fig.colorbar(im, cax=cax, orientation='vertical')
                plt.suptitle('NDVI')
                ax.axis('off')
                fig.tight_layout(pad=0)
                ax.margins(0)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                self.plants[plant_name]['ndvi_image'] = (image_from_plot, 'NDVI Image')
                self.plant_stats[plant_name]['NDVI (Maximum)'] = max_ndvi
                self.plant_stats[plant_name]['NDVI (Minimum)'] = min_ndvi
                self.plant_stats[plant_name]['NDVI (Average)'] = avg_ndvi
                self.plant_stats[plant_name]['NDVI (Positive Average)'] = pos_avg_ndvi

    
    def stitch_color_images(self, batch):
        
        for plant_name in batch:
            
            input_images = [color_image for color_image,image_name in self.plants[plant_name]['color_images']]
            stitched_image = image_stitching(input_images)
            self.plants[plant_name]['stitched_image'] = (stitched_image, 'Whole Plant Image')
            
    def calculate_connected_components(self, batch):
        
        for plant_name in batch:
            
            gray_image, binary = CCA_Preprocess(self.plants[plant_name]['stitched_image'][0], k = self.variable_k)
            preprocessed_image = np.repeat(np.expand_dims(binary, axis=-1), 3, axis=-1) * self.plants[plant_name]['stitched_image'][0]
            cca_image = 255*(preprocessed_image - preprocessed_image.min()) / (preprocessed_image.max() - preprocessed_image.min())
            cca_image = cca_image.astype(np.uint8)
            self.plants[plant_name]['cca_image'] = (cca_image, 'Background Separated Using Connected Component Analysis')
        
    def run_segmentation(self, batch):
        
        input_images, plant_names = [self.plants[plant_name]['stitched_image'][0] for plant_name in batch], batch
        results = self.segmentation_model.predict(input_images, conf = pipeline_config['segmentation_confidence'], device = self.device)
        
        for result_index in range(len(results)):
            
            result = results[result_index]
            
            if result:
            
                if result.masks.data.shape[0] > 4:
                    result.masks.data = result.masks.data[:4]
                #print(plant_names[result_index],result.masks.data.shape)
                mask = preprocess_mask(result.masks.data)
                binary_mask_np = generate_binary_mask(mask)
                overlayed_image = overlay_mask_on_image(binary_mask_np, self.plants[plant_names[result_index]]['stitched_image'][0])
                self.plants[plant_names[result_index]]['segmented_image'] = (overlayed_image, 'Background Separated Using Image Segmentation')
    
    def calculate_plant_phenotypes(self, batch):

        for plant_name in batch:

            phenotypes = get_plant_phenotypes(self.plants[plant_name]['segmented_image'][0], offset = self.offset)
            self.plant_stats[plant_name]['Height'] = phenotypes['Plant Height (cm)']
            self.plant_stats[plant_name]['Width'] = phenotypes['Plant Width (cm)']
            self.plant_stats[plant_name]['Area'] = phenotypes['Plant Area (square cm)']
            self.plant_stats[plant_name]['Perimeter'] = phenotypes['Plant Perimeter (cm)']
            self.plant_stats[plant_name]['Solidity'] = phenotypes['Plant Solidity']
            self.plant_stats[plant_name]['Number of Branches'] = phenotypes['Number of Branches']
            self.plant_stats[plant_name]['Number of Leaves'] = phenotypes['Number of Leaves']
        
    def calculate_tips_and_branches(self, batch):

        for plant_name in batch:

            # pcv.outputs.clear()
            gray_image = cv2.cvtColor(self.plants[plant_name]['segmented_image'][0], cv2.COLOR_RGB2GRAY)
            skeleton = pcv.morphology.skeletonize(mask = gray_image)
            tips = pcv.morphology.find_tips(skel_img = skeleton, mask = None, label = plant_name)
            branches = pcv.morphology.find_branch_pts(skel_img = skeleton, mask = None, label = plant_name)
            tips_and_branches = np.zeros_like(skeleton)
            tips_and_branches[tips > 0] = 255
            tips_and_branches[branches > 0] = 128
            kernel = np.ones((5, 5), np.uint8)
            tips = cv2.dilate(tips, kernel, iterations = 1)
            branches = cv2.dilate(branches, kernel, iterations = 1)
            tips_and_branches = cv2.dilate(tips_and_branches, kernel, iterations = 1)
            self.plants[plant_name]['tips'] = (tips, 'Plant Tips')
            self.plants[plant_name]['branches'] = (branches, 'Plant Branch Points')
            self.plants[plant_name]['tips_and_branches'] = (tips_and_branches, 'Plant Tips and Branch Points')
            self.plants[plant_name]['gray_image'] = (gray_image, 'Gray Segmented Image')
            self.plants[plant_name]['skeleton'] = (skeleton, 'Morphology Skeleton')
        
    def calculate_sift_features(self, batch):

        for plant_name in batch:

            sift = cv2.SIFT_create()
            kp, des= sift.detectAndCompute(self.plants[plant_name]['skeleton'][0], None)
            sift_image = cv2.drawKeypoints(self.plants[plant_name]['skeleton'][0], kp, des)
            self.plants[plant_name]['sift_features'] = (sift_image, 'SIFT Features')
        
    def calculate_LBP_features(self, batch):

        for plant_name in batch:

            lbp = local_binary_pattern(self.plants[plant_name]['gray_image'][0], self.LBP_n_points, self.LBP_radius)
            self.plants[plant_name]['lbp_features'] = (lbp, 'Local Binary Patterns')
        
    def calculate_HOG_features(self, batch):

        for plant_name in batch:

            fd,hog_image = hog(self.plants[plant_name]['gray_image'][0], orientations = pipeline_config['HOG_orientations'], pixels_per_cell = (pipeline_config['HOG_pixels_per_cell'], pipeline_config['HOG_pixels_per_cell']), cells_per_block = (pipeline_config['HOG_cells_per_block'], pipeline_config['HOG_cells_per_block']), visualize=True, multichannel=False, channel_axis=-1)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, pipeline_config['HOG_orientations']))
            hog_image_rescaled = hog_image_rescaled*255
            self.plants[plant_name]['hog_features'] = (hog_image_rescaled, 'Histogram of Oriented Gradients')

    def clear(self):

        self.service_type = 0
        self.input_folder_path = None
        self.output_folder_path = None
        self.show_raw_images = False
        self.show_color_images = False
        del self.plant_paths
        del self.plant_stats
        shutil.rmtree(self.interm_result_folder)

    def reset(self):

        print('session reset')

    def get_plant_statistics_df(self):

        plant_names = self.get_plant_names()
        df_dict = {}
        df_dict['Plant_Name'] = plant_names

        for item in self.statistics_items:

            df_dict[item] = [round(self.plant_stats[plant_name][item],2) for plant_name in plant_names]
        
        return pd.DataFrame(df_dict)

    def make_dir(self, folder):

        if not os.path.exists(folder):

            os.mkdir(folder)
   
    def save_results(self, folder_path):

        self.make_dir(folder_path)
        
        result_dict = {}
        result_dict['statistics_items'] = self.statistics_items
        result_dict['statistics_units'] = self.statistics_units
        
        filepath = os.path.join(folder_path, 'plants_features_and_statistics.txt')
        f = open(filepath, 'w')
        outputs = pcv.outputs.observations
        plant_names = self.get_plant_names()
        
        for plant_name in plant_names:

            if plant_name not in result_dict.keys():
                result_dict[plant_name] = {}
        
            tips_list = outputs[plant_name]['tips']['value']
            branch_pts_list = outputs[plant_name]['branch_pts']['value']
            line = plant_name+',tips,'+','.join([str(coord[0])+','+str(coord[1]) for coord in tips_list])+'\n'
            f.write(line)
            line = plant_name+',branch_points,'+','.join([str(coord[0])+','+str(coord[1]) for coord in branch_pts_list])+'\n'
            f.write(line)
            result_dict[plant_name]['tips'] = tips_list
            result_dict[plant_name]['branch_points'] = branch_pts_list

            for item in self.statistics_items:
                
                line = plant_name+','+item+','+str(self.plant_stats[plant_name][item])+'\n'
                f.write(line)
                result_dict[plant_name][item] = self.plant_stats[plant_name][item]
        
        f.close()
        
        json_filepath = os.path.join(folder_path, 'plant_features_and_statistics.json')
        
        with open(json_filepath, 'w') as fp:
            
            json.dump(result_dict, fp, indent = 4)

        for plant_name in plant_names:

            plant_folder = os.path.join(folder_path, plant_name)
            self.make_dir(plant_folder)
            color_images_folder = os.path.join(plant_folder, 'Color_Images')
            self.make_dir(color_images_folder)

            color_images = self.get_color_images(plant_name)
            
            for image, image_name in color_images:

                image_name = image_name.split('.')[0]+'.jpg'
                cv2.imwrite(os.path.join(color_images_folder,image_name),image)

            analysis_images = self.get_plant_analysis_images(plant_name)
            
            for image,name in analysis_images:

                cv2.imwrite(os.path.join(plant_folder,'_'.join(name.split(' '))+'.jpg'), image)
    
