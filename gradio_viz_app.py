import gradio as gr
import json
import os
import pandas as pd
import yaml
import cv2

class GUI_Viz():

    def __init__(self):
        
        self.head = (
                    "<center>"
                    "<a href='https://precisiongreenhouse.tamu.edu/'><img src='https://peepleslab.engr.tamu.edu/wp-content/uploads/sites/268/2023/04/AgriLife_Logo-e1681857158121.png' width=1650></a>"
                    "<br>"
                    "Visualization of Plant Analysis Results"
                    "<br>"
                    "<a href ='https://precisiongreenhouse.tamu.edu/'>The Texas A&M Plant Growth and Phenotyping Facility Data Analysis Pipeline</a>"
                    "</center>"
                )

        self.theme = gr.themes.Base(
                primary_hue="violet",
                secondary_hue="green",).set(body_background_fill_dark='*checkbox_label_background_fill')
        self.demo = gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.lime))
        self.statistics_items = ['Height', 'Width', 'Area', 'Perimeter', 'Solidity', 'Number of Branches', 'Number of Leaves', 'NDVI (Maximum)', 'NDVI (Minimum)', 'NDVI (Average)', 'NDVI (Positive Average)']
        self.statistics_units = [' cm', ' cm', ' square cm', ' cm', '', '', '', '', '', '', '']
        self.summary_statistics_tabs = {}
        self.summary_plots = {}
        self.summary_galleries = {}
        
        with self.demo:

            gr.HTML(value = self.head)

            with gr.Column():

                self.folderpath_input = gr.Textbox(label = 'Enter folder path of results directory',
                                            show_label = True,
                                            type = 'text',
                                            visible = True)

                self.visualize_button = gr.Button(value = 'Visualize',
                                          visible = True)

                with gr.Tabs():
    
                    for item in self.statistics_items:
    
                        self.summary_statistics_tabs[item] = gr.Tab(label = 'Height', visible = False)
    
                    for item in self.statistics_items:
    
                        with self.summary_statistics_tabs[item]:
    
                            with gr.Row():
    
                                self.summary_plots[item] = gr.BarPlot(visible = False)
                                self.summary_galleries[item] = gr.Gallery(visible = False)

                self.plant_select_dropdown = gr.Dropdown( multiselect = False, 
                                                       label = 'Select Plant',
                                                       show_label = True, 
                                                       visible = False,
                                                       type = 'value')

                with gr.Tabs():
                    
                    self.color_images_tab = gr.Tab(label = 'Color Input Images', visible = False)
                    self.plant_analysis_tab = gr.Tab(label = 'Plant Analysis', visible = False)
                    self.plant_statistics_tab = gr.Tab(label = 'Plant Statistics', visible = False)
                    
                    with self.color_images_tab:
                        
                        self.color_images_gallery = gr.Gallery(label = 'Color Input Images',
                                                    show_label = True,
                                                    height = 512,
                                                    preview = True,
                                                    visible = False)
                    
                    with self.plant_analysis_tab:
                        
                        self.plant_analysis_gallery = gr.Gallery(label = 'Plant Analysis',
                                                    show_label = True,
                                                    height = 512,
                                                    preview = True,
                                                    visible = False)
                    
                    with self.plant_statistics_tab:
                        
                        self.plant_statistics_df = gr.Dataframe(label = 'Plant Phenotypic Traits',
                                             show_label = True,
                                             visible = False)
                
                self.refresh_button = gr.Button(value = 'Refresh',
                                          visible = False)

            self.visualize_button.click(self.analyze_results,
                                       inputs = [self.folderpath_input],
                                       outputs = [self.summary_statistics_tabs[item] for item in self.statistics_items]+[self.summary_plots[item] for item in self.statistics_items]+[self.summary_galleries[item] for item in self.statistics_items]+[self.plant_select_dropdown,self.refresh_button])    

            self.plant_select_dropdown.input(self.show_plant_analysis_result,
                                             inputs = [self.plant_select_dropdown],
                                             outputs = [self.color_images_tab, 
                                                        self.plant_analysis_tab, 
                                                        self.plant_statistics_tab,
                                                        self.color_images_gallery,
                                                        self.plant_analysis_gallery,
                                                        self.plant_statistics_df])
            
            self.refresh_button.click(self.reset,
                                      js="window.location.reload()")
            
    def analyze_results(self, folder_path):

        self.result_dir_path = folder_path
        
        json_file_path = os.path.join(folder_path, 'plant_features_and_statistics.json')
        
        with open(json_file_path, 'r') as fp:
            self.results_dict = json.load(fp)

        del self.results_dict['statistics_items']
        del self.results_dict['statistics_units']

        self.plant_names = sorted(list(self.results_dict.keys()))

        outputs = []
        
        for item in self.statistics_items:

            outputs.append(gr.Tab(label = item, visible = True))

        plant_statistics_df = self.get_plant_statistics_df()

        for index,item in enumerate(self.statistics_items):

            outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = item,
                                  title = item,
                                  tooltip = item,
                                  x_title = 'Plant',
                                  y_title = item+' '+self.statistics_units[index],
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant' + item + 'Plot',
                                  show_label = True,
                                  visible = True))

        min_indices = plant_statistics_df.idxmin(numeric_only = True)
        max_indices = plant_statistics_df.idxmax(numeric_only = True)
        
        for statistic in self.statistics_items:
            
            outputs.append(gr.Gallery(value = [(self.get_segmented_image(self.plant_names[min_indices[statistic]]),'Minimum Value: ' + self.plant_names[min_indices[statistic]]), (self.get_segmented_image(self.plant_names[max_indices[statistic]]),'Maximum Value: ' + self.plant_names[max_indices[statistic]])],
                                label = statistic+' Comparison',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = True))
        
        outputs.append(gr.Dropdown(choices = self.plant_names,
                           multiselect = False, 
                           label = 'Select Plant',
                           show_label = True, 
                           visible = True,
                           type = 'value'))
        outputs.append(gr.Button(value = 'Refresh',
                                visible = True))
        
        return outputs
    
    def get_plant_statistics_df(self):

        df_dict = {}
        df_dict['Plant_Name'] = self.plant_names

        for item in self.statistics_items:

            df_dict[item] = [round(self.results_dict[plant_name][item],2) for plant_name in self.plant_names]
        
        return pd.DataFrame(df_dict)

    def get_plant_statistics_df_plantwise(self, plant):
        
        return pd.DataFrame({'Phenotypic trait': self.statistics_items,
                             'Value': [str(round(self.results_dict[plant][self.statistics_items[index]],2))+self.statistics_units[index] for index in range(len(self.statistics_items))]})
    
    def show_plant_analysis_result(self, plant):

        outputs = []
        outputs.append(gr.Tab(label = 'Color Images', visible = True))
        outputs.append(gr.Tab(label = 'Plant Analysis', visible = True))
        outputs.append(gr.Tab(label = 'Plant Statistics', visible = True))
        outputs.append(gr.Gallery(value = self.get_color_images_for_gallery(plant),
                                label = 'Color Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = True))
        outputs.append(gr.Gallery(value = self.get_analysis_images_for_gallery(plant),
                                label = 'Plant Analysis Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = True))
        outputs.append(gr.Dataframe(value = self.get_plant_statistics_df_plantwise(plant),
                                 label = 'Estimated Plant Phenotypic Traits are ',
                                 show_label = True,
                                 visible = True))
        return outputs

    def get_segmented_image(self, plant):

        return cv2.imread(os.path.join(self.result_dir_path,plant,'Background_Separated_Using_Image_Segmentation.jpg'))

    def get_analysis_images_for_gallery(self, plant):

        plant_folder_path = os.path.join(self.result_dir_path, plant)
        images_list = sorted(os.listdir(plant_folder_path))
        images_list.remove('Color_Images')
        if '.ipynb_checkpoints' in images_list:
            images_list.remove('.ipynb_checkpoints')
        return [(cv2.imread(os.path.join(plant_folder_path,image_name)),image_name.split('.')[0]) for image_name in images_list]
        
    def get_color_images_for_gallery(self, plant):

        color_images_folder_path = os.path.join(self.result_dir_path, plant, 'Color_Images')
        images_list = sorted(os.listdir(color_images_folder_path))
        if '.ipynb_checkpoints' in images_list:
            images_list.remove('.ipynb_checkpoints')
        return [(cv2.imread(os.path.join(color_images_folder_path,image_name)),image_name.split('.')[0]) for image_name in images_list]

    def reset(self):
        
        print('Reset')

gui_viz = GUI_Viz()
demo = gui_viz.demo
demo.launch(share = True)
