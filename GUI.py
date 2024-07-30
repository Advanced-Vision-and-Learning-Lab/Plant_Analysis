from Plant_Analysis import Plant_Analysis
import gradio as gr
from time import time
import yaml

file = open('pipeline_config.yaml', 'r')
pipeline_config = yaml.safe_load(file)
file.close()

'''import sys

class Logger:

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

log_filename = 'GUI_output.log'

sys.stdout = Logger(log_filename)

def read_logs():
    
    sys.stdout.write('Gradio Application\n')
    sys.stdout.flush()
    with open(log_filename, "r") as f:
        return f.read()'''


class GUI():

    def __init__(self):

        self.session_index = 1
        self.device = pipeline_config['device']

        self.head = (
                    "<center>"
                    "<a href='https://precisiongreenhouse.tamu.edu/'><img src='https://peepleslab.engr.tamu.edu/wp-content/uploads/sites/268/2023/04/AgriLife_Logo-e1681857158121.png' width=1650></a>"
                    "<br>"
                    "Plant Analysis and Feature Extraction Demonstration"
                    "<br>"
                    "<a href ='https://precisiongreenhouse.tamu.edu/'>The Texas A&M Plant Growth and Phenotyping Facility Data Analysis Pipeline</a>"
                    "</center>"
                )
        

        self.theme = gr.themes.Base(
                primary_hue="violet",
                secondary_hue="green",).set(body_background_fill_dark='*checkbox_label_background_fill')
        self.demo = gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.lime))
        self.service_dropdown_choices = ['Multi Plant Analysis', 'Single Plant Analysis']
        self.plant_analysis = {}
        
        with self.demo:

            #self.demo.load(read_logs, None, None, every=1)

            self.session_name = gr.State([])
            
            gr.HTML(value = self.head)
            
            with gr.Column():
                
                self.service_dropdown = gr.Dropdown(choices = self.service_dropdown_choices, 
                                       multiselect = False, 
                                       label = 'Select Service',
                                       show_label = True, 
                                       visible = True,
                                       type = 'index')
                
                self.filepath_input = gr.Textbox(label = 'Enter folder path containing plant images',
                                            show_label = True,
                                            type = 'text',
                                            visible = False)

                self.input_submit_button = gr.Button(value = 'Submit Input Folder Path',
                                          visible = False)
                
                with gr.Row():

                    self.show_input_checkbox = gr.Checkbox(label = 'Show Raw Images',
                                                      info = 'Check to show input images',
                                                      value = False,
                                                      visible = False)
                    
                    self.show_color_images_checkbox = gr.Checkbox(label = 'Show Color Images',
                                                      info = 'Check to show color images',
                                                      value = False,
                                                      visible = False)
                    
                self.request_submit_button = gr.Button(value = 'Submit',
                                                       visible = False)
                
                self.pre_information_textbox = gr.Textbox(label = 'Information',
                                                        visible = False)

                self.post_information_textbox = gr.Textbox(label = 'Information',
                                                        visible = False)                

                #outputs

                with gr.Tabs():
                    
                    self.plant_height_tab = gr.Tab(label = 'Height', visible = False)
                    self.plant_width_tab = gr.Tab(label = 'Width', visible = False)
                    self.plant_area_tab = gr.Tab(label = 'Area', visible = False)
                    self.plant_perimeter_tab = gr.Tab(label = 'Perimeter', visible = False)
                    self.plant_solidity_tab = gr.Tab(label = 'Solidity', visible = False)
                    self.plant_branches_tab = gr.Tab(label = 'Number of Branches', visible = False)
                    self.plant_leaves_tab = gr.Tab(label = 'Number of Leaves', visible = False)
                    self.plant_ndvi_max_tab = gr.Tab(label = 'NDVI (Maximum)', visible = False)
                    self.plant_ndvi_min_tab = gr.Tab(label = 'NDVI (Minimum)', visible = False)
                    self.plant_ndvi_avg_tab = gr.Tab(label = 'NDVI (Average)', visible = False)
                    self.plant_ndvi_pos_avg_tab = gr.Tab(label = 'NDVI (Positive Average)', visible = False)
                    # Add NDVI, Solidity, Perimeter
                    
                    with self.plant_height_tab:
                        
                        with gr.Row():

                            self.plant_height_plot = gr.BarPlot(visible = False)
                            self.plant_height_gallery = gr.Gallery(visible = False)

                    with self.plant_width_tab:
                        
                        with gr.Row():

                            self.plant_width_plot = gr.BarPlot(visible = False)
                            self.plant_width_gallery = gr.Gallery(visible = False)

                    with self.plant_area_tab:
                        
                        with gr.Row():

                            self.plant_area_plot = gr.BarPlot(visible = False)
                            self.plant_area_gallery = gr.Gallery(visible = False)

                    with self.plant_perimeter_tab:

                        with gr.Row():

                            self.plant_perimeter_plot = gr.BarPlot(visible = False)
                            self.plant_perimeter_gallery = gr.Gallery(visible = False)

                    with self.plant_solidity_tab:

                        with gr.Row():

                            self.plant_solidity_plot = gr.BarPlot(visible = False)
                            self.plant_solidity_gallery = gr.Gallery(visible = False)

                    with self.plant_branches_tab:

                        with gr.Row():

                            self.plant_branches_plot = gr.BarPlot(visible = False)
                            self.plant_branches_gallery = gr.Gallery(visible = False)

                    with self.plant_leaves_tab:

                        with gr.Row():

                            self.plant_leaves_plot = gr.BarPlot(visible = False)
                            self.plant_leaves_gallery = gr.Gallery(visible = False)

                    with self.plant_ndvi_max_tab:
                        
                        with gr.Row():

                            self.plant_ndvi_max_plot = gr.BarPlot(visible = False)
                            self.plant_ndvi_max_gallery = gr.Gallery(visible = False)

                    with self.plant_ndvi_min_tab:
                        
                        with gr.Row():

                            self.plant_ndvi_min_plot = gr.BarPlot(visible = False)
                            self.plant_ndvi_min_gallery = gr.Gallery(visible = False)

                    with self.plant_ndvi_avg_tab:

                        with gr.Row():

                            self.plant_ndvi_avg_plot = gr.BarPlot(visible = False)
                            self.plant_ndvi_avg_gallery = gr.Gallery(visible = False)

                    with self.plant_ndvi_pos_avg_tab:

                        with gr.Row():

                            self.plant_ndvi_pos_avg_plot = gr.BarPlot(visible = False)
                            self.plant_ndvi_pos_avg_gallery = gr.Gallery(visible = False)
                
                self.plant_select_dropdown = gr.Dropdown( multiselect = False, 
                                                       label = 'Select Plant',
                                                       show_label = True, 
                                                       visible = False,
                                                       type = 'value')
                
                with gr.Tabs():
                    
                    self.input_images_tab = gr.Tab(label = 'Raw Input Images', visible = False)
                    self.color_images_tab = gr.Tab(label = 'Color Input Images', visible = False)
                    self.plant_analysis_tab = gr.Tab(label = 'Plant Analysis', visible = False)
                    self.plant_statistics_tab = gr.Tab(label = 'Plant Statistics', visible = False)
                    
                    with self.input_images_tab:
                        
                        self.input_images_gallery = gr.Gallery(label = 'Uploaded Raw Input Images',
                                                    show_label = True,
                                                    height = 512,
                                                    preview = True,
                                                    visible = False)
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

                self.output_folder_textbox = gr.Textbox(label = 'Enter path to save results to',
                                                       show_label = True,
                                                       visible = False)

                self.save_result_button = gr.Button(value = 'SAVE RESULTS',
                                                   visible = False)

                self.saving_information_textbox = gr.Textbox(label = 'Information',
                                                            show_label = False,
                                                            visible = False)

                self.clear_button = gr.Button(value = 'CLEAR CACHE',
                                                   visible = False)


                self.clear_info_textbox = gr.Textbox(label = 'Information',
                                                    show_label = False,
                                                    visible = False)

                self.reset_button = gr.Button(value = 'REFRESH',
                                               visible = False)
            
                # self.reset_button = gr.ClearButton(components = [self.filepath_input,
                #                                 self.show_input_checkbox,
                #                                 self.show_color_images_checkbox,
                #                                 self.pre_information_textbox,
                #                                 self.post_information_textbox,
                #                                 self.plant_select_dropdown,
                #                                 self.input_images_tab,
                #                                 self.input_images_gallery,
                #                                 self.color_images_tab,
                #                                 self.color_images_gallery,
                #                                 self.plant_analysis_tab,
                #                                 self.plant_analysis_gallery,
                #                                 self.plant_statistics_tab,
                #                                 self.plant_statistics_df,
                #                                 self.output_folder_textbox,
                #                                 self.saving_information_textbox],
                #                                 value = 'CLEAR',
                #                                 visible = False)
                    
        
            self.service_dropdown.input(self.update_service,
                                        inputs = [self.session_name,
                                                  self.service_dropdown],
                                        outputs = [self.filepath_input,
                                                   self.input_submit_button,
                                                   self.session_name])

            self.input_submit_button.click(self.update_input_path,
                                           inputs = [self.session_name,
                                                    self.filepath_input],
                                           outputs = [self.show_input_checkbox,
                                                      self.show_color_images_checkbox,
                                                      self.request_submit_button])

            self.show_input_checkbox.input(self.update_check_RI_option,
                                           inputs = [self.session_name,
                                                    self.show_input_checkbox])

            self.show_color_images_checkbox.input(self.update_check_CI_option,
                                                  inputs = [self.session_name,
                                                            self.show_color_images_checkbox])
            
            self.request_submit_button.click(self.update_info_textbox,
                                             inputs = self.session_name,
                                             outputs = self.pre_information_textbox)

            self.request_submit_button.click(self.get_plant_analysis,
                                             inputs = self.session_name,
                                             outputs = [self.post_information_textbox,
                                                        self.plant_height_tab,
                                                        self.plant_width_tab,
                                                        self.plant_area_tab,
                                                        self.plant_perimeter_tab,
                                                        self.plant_solidity_tab,
                                                        self.plant_branches_tab,
                                                        self.plant_leaves_tab,
                                                        self.plant_ndvi_max_tab,
                                                        self.plant_ndvi_min_tab,
                                                        self.plant_ndvi_avg_tab,
                                                        self.plant_ndvi_pos_avg_tab,
                                                        self.plant_height_plot,
                                                        self.plant_width_plot,
                                                        self.plant_area_plot,
                                                        self.plant_perimeter_plot,
                                                        self.plant_solidity_plot,
                                                        self.plant_branches_plot,
                                                        self.plant_leaves_plot,
                                                        self.plant_ndvi_max_plot,
                                                        self.plant_ndvi_min_plot,
                                                        self.plant_ndvi_avg_plot,
                                                        self.plant_ndvi_pos_avg_plot,
                                                        self.plant_height_gallery,
                                                        self.plant_width_gallery,
                                                        self.plant_area_gallery,
                                                        self.plant_perimeter_gallery,
                                                        self.plant_solidity_gallery,
                                                        self.plant_branches_gallery,
                                                        self.plant_leaves_gallery,
                                                        self.plant_ndvi_max_gallery,
                                                        self.plant_ndvi_min_gallery,
                                                        self.plant_ndvi_avg_gallery,
                                                        self.plant_ndvi_pos_avg_gallery,
                                                        self.plant_select_dropdown])
            
            self.plant_select_dropdown.input(self.show_plant_analysis_result,
                                             inputs = [self.session_name,
                                                       self.plant_select_dropdown],
                                             outputs = [self.input_images_tab, 
                                                        self.color_images_tab, 
                                                        self.plant_analysis_tab, 
                                                        self.plant_statistics_tab,
                                                        self.input_images_gallery,
                                                        self.color_images_gallery,
                                                        self.plant_analysis_gallery,
                                                        self.plant_statistics_df,
                                                        self.output_folder_textbox,
                                                        self.save_result_button,
                                                        self.clear_button])

            self.save_result_button.click(self.save_analysis_result,
                                         inputs = [self.session_name,
                                                   self.output_folder_textbox],
                                         outputs = self.saving_information_textbox)

            self.clear_button.click(self.clear, inputs = self.session_name, outputs = [self.clear_info_textbox,
                                                                                       self.reset_button])
            
            self.reset_button.click(self.reset, inputs = self.session_name,  js="window.location.reload()")
            
    
    def update_service(self, session, service_type):
        
        session.append('session_'+str(self.session_index))
        self.plant_analysis[session[0]] = Plant_Analysis(session = session[0])
        self.session_index += 1
        if self.session_index == 100000:
            self.session_index = 1

        self.plant_analysis[session[0]].update_service_type(service_type)
        outputs = []
        outputs.append(gr.Textbox(label = 'Enter folder path containing plant images',
                                            show_label = True,
                                            type = 'text',
                                            visible = True))
        outputs.append(gr.Button(value = 'Submit Input Folder Path',
                                          visible = True))
        outputs.append(session)
        return outputs
        
    def update_input_path(self, session, input_path):

        self.plant_analysis[session[0]].update_input_path(input_path)
        outputs = []
        outputs.append(gr.Checkbox(label = 'Show Raw Images',
                                                      info = 'Check to show input images',
                                                      value = False,
                                                      visible = True))
        outputs.append(gr.Checkbox(label = 'Show Color Images',
                                                      info = 'Check to show color images',
                                                      value = False,
                                                      visible = True))
        outputs.append(gr.Button(value = 'Submit',
                                          visible = True))
        return outputs

    def update_check_RI_option(self, session,  check_RI):

        self.plant_analysis[session[0]].update_check_RI_option(check_RI)

    def update_check_CI_option(self, session, check_CI):

        self.plant_analysis[session[0]].update_check_CI_option(check_CI)

    def update_info_textbox(self, session):

        information = 'Request Submitted. Processing ' + str(len(self.plant_analysis[session[0]].get_plant_names())) + ' Plants.\n\nPlease wait..'

        return gr.Textbox(show_label = False,
                          value = information,
                          visible = True)

    def get_plant_analysis(self, session):

        # self.plant_analysis[session[0]].make_color_images()
        # self.plant_analysis[session[0]].stitch_color_images()
        # self.plant_analysis[session[0]].calculate_connected_components()
        self.plant_analysis[session[0]].load_segmentation_model()
        # self.plant_analysis[session[0]].run_segmentation()
        # self.plant_analysis[session[0]].calculate_plant_phenotypes()
        # self.plant_analysis[session[0]].calculate_tips_and_branches()
        # self.plant_analysis[session[0]].calculate_sift_features()
        # self.plant_analysis[session[0]].calculate_LBP_features()
        # self.plant_analysis[session[0]].calculate_HOG_features()
        self.plant_analysis[session[0]].do_plant_analysis()
        information = 'Processing Complete.\n\nSelect plant from the dropdown below to check the output of individual plants'

        outputs = []

        outputs.append(gr.Textbox(show_label = False,
                                  value = information,
                                  visible = True))
        outputs.append(gr.Tab(label = 'Height', visible = True))
        outputs.append(gr.Tab(label = 'Width', visible = True))
        outputs.append(gr.Tab(label = 'Area', visible = True))
        outputs.append(gr.Tab(label = 'Perimeter', visible = True))
        outputs.append(gr.Tab(label = 'Solidity', visible = True))
        outputs.append(gr.Tab(label = 'Number of Branches', visible = True))
        outputs.append(gr.Tab(label = 'Number of Leaves', visible = True))
        outputs.append(gr.Tab(label = 'NDVI (Maximum)', visible = True))
        outputs.append(gr.Tab(label = 'NDVI (Minimum)', visible = True))
        outputs.append(gr.Tab(label = 'NDVI (Average)', visible = True))
        outputs.append(gr.Tab(label = 'NDVI (Positive Average)', visible = True))
        plant_statistics_df = self.plant_analysis[session[0]].get_plant_statistics_df()
        num_plants = len(list(plant_statistics_df['Plant_Name']))
        plant_names = self.plant_analysis[session[0]].get_plant_names()
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Height',
                                  title = 'Plant Height',
                                  tooltip = 'Height',
                                  x_title = 'Plant',
                                  y_title = 'Height (cm)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Height Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Width',
                                  title = 'Plant Width',
                                  tooltip = 'Width',
                                  x_title = 'Plant',
                                  y_title = 'Width (cm)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Width Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Area',
                                  title = 'Plant Area',
                                  tooltip = 'Area',
                                  x_title = 'Plant',
                                  y_title = 'Area (square cm)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Area Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Perimeter',
                                  title = 'Plant Perimeter',
                                  tooltip = 'Perimeter',
                                  x_title = 'Plant',
                                  y_title = 'Perimeter (cm)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Perimeter Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Solidity',
                                  title = 'Plant Solidity',
                                  tooltip = 'Solidity',
                                  x_title = 'Plant',
                                  y_title = 'Solidity',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Solidity Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Number of Branches',
                                  title = 'Number of Branches',
                                  tooltip = 'Number of Branches',
                                  x_title = 'Plant',
                                  y_title = 'Number of Branches',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Branch Count Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Number of Leaves',
                                  title = 'Number of Leaves',
                                  tooltip = 'Number of Leaves',
                                  x_title = 'Plant',
                                  y_title = 'Number of Leaves',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant Leaf Count Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'NDVI (Maximum)',
                                  title = 'NDVI (Maximum)',
                                  tooltip = 'NDVI (Maximum)',
                                  x_title = 'Plant',
                                  y_title = 'NDVI (Maximum)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant NDVI (Maximum) Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'NDVI (Minimum)',
                                  title = 'NDVI (Minimum)',
                                  tooltip = 'NDVI (Minimum)',
                                  x_title = 'Plant',
                                  y_title = 'NDVI (Minimum)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant NDVI (Minimum) Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'NDVI (Average)',
                                  title = 'NDVI (Average)',
                                  tooltip = 'NDVI (Average)',
                                  x_title = 'Plant',
                                  y_title = 'NDVI (Average)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant NDVI (Average) Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'NDVI (Positive Average)',
                                  title = 'NDVI (Positive Average)',
                                  tooltip = 'NDVI (Positive Average)',
                                  x_title = 'Plant',
                                  y_title = 'NDVI (Positive Average)',
                                  x_label_angle = 0,
                                  y_label_angle = 0,
                                  vertical = False,
                                  width = 400,
                                  label = 'Plant NDVI (Positive Average) Plot',
                                  show_label = True,
                                  visible = True))
        
        #print(plant_statistics_df)

        min_indices = plant_statistics_df.idxmin(numeric_only = True)
        max_indices = plant_statistics_df.idxmax(numeric_only = True)
        for statistic in self.plant_analysis[session[0]].statistics_items:
            outputs.append(gr.Gallery(value = [(self.plant_analysis[session[0]].get_segmented_image(plant_names[min_indices[statistic]]),'Minimum Value: '+plant_names[min_indices[statistic]]),(self.plant_analysis[session[0]].get_segmented_image(plant_names[max_indices[statistic]]),'Maximum Value: '+plant_names[max_indices[statistic]])],
                                label = statistic+' Comparison',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = True))
        outputs.append(gr.Dropdown(choices = self.plant_analysis[session[0]].get_plant_names(),
                           multiselect = False, 
                           label = 'Select Plant',
                           show_label = True, 
                           visible = True,
                           type = 'value'))

        return outputs

    def show_plant_analysis_result(self, session, plant):

        outputs = []

        outputs.append(gr.Tab(label = 'Raw Input Images', visible = self.plant_analysis[session[0]].show_raw_images))
        outputs.append(gr.Tab(label = 'Color Input Images', visible = self.plant_analysis[session[0]].show_color_images))
        outputs.append(gr.Tab(label = 'Plant Analysis', visible = True))
        outputs.append(gr.Tab(label = 'Plant Statistics', visible = True))
        outputs.append(gr.Gallery(value = self.plant_analysis[session[0]].get_raw_images(plant) if self.plant_analysis[session[0]].show_raw_images else [],
                                label = 'Uploaded Raw Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = self.plant_analysis[session[0]].show_raw_images))
        outputs.append(gr.Gallery(value = self.plant_analysis[session[0]].get_color_images(plant) if self.plant_analysis[session[0]].show_color_images else [],
                                label = 'Color Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = self.plant_analysis[session[0]].show_color_images))
        outputs.append(gr.Gallery(value = self.plant_analysis[session[0]].get_plant_analysis_images(plant),
                                label = 'Plant Analysis',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = True))
        outputs.append(gr.Dataframe(value = self.plant_analysis[session[0]].get_plant_statistics_df_plantwise(plant),
                                 label = 'Estimated Plant Phenotypic Traits are ',
                                 show_label = True,
                                 visible = True))
        outputs.append(gr.Textbox(label = 'Enter path to save results to',
                               show_label = True,
                               value = 'Results',
                               visible = True))
        outputs.append(gr.Button(value = 'SAVE RESULTS',
                               visible = True))
        outputs.append(gr.Button(value = 'CLEAR CACHE',
                               visible = True))

        return outputs

    def save_analysis_result(self, session, output_folder_path):

        self.plant_analysis[session[0]].save_results(output_folder_path)

        return gr.Textbox(label = 'Information',
                         show_label = False,
                         value = 'Saved Result to the path: ' + output_folder_path,
                         visible = True)

    def clear(self, session):

        self.plant_analysis[session[0]].clear()
        outputs = []
        
        outputs.append(gr.Textbox(label = 'Information',
                                show_label = False,
                                value = 'Cleared Cache. Click REFRESH button for a new session',
                                visible = True))

        outputs.append(gr.Button(value = 'REFRESH',
                                visible = True))

        return outputs
    
    def reset(self, session):

        self.plant_analysis[session[0]].reset()
        del self.plant_analysis[session[0]]
        print(session[0] + ' is cleared')

    def reset_depricated(self):

        self.plant_analysis.reset()
        
        outputs = []
        
        outputs.append(gr.Textbox(label = 'Enter folder path containing plant images',
                                            show_label = True,
                                            type = 'text',
                                            visible = False))
        outputs.append(gr.Button(value = 'Submit Input Folder Path',
                                          visible = False))
        outputs.append(gr.Checkbox(label = 'Show Raw Images',
                                                      info = 'Check to show input images',
                                                      value = False,
                                                      visible = False))
        outputs.append(gr.Checkbox(label = 'Show Color Images',
                                                      info = 'Check to show color images',
                                                      value = False,
                                                      visible = False))
        outputs.append(gr.Button(value = 'Submit',
                                          visible = False))
        outputs.append(gr.Dropdown( multiselect = False, 
                                   label = 'Select Plant',
                                   show_label = True, 
                                   visible = False,
                                   type = 'value'))
        outputs.append(gr.Tab(label = 'Raw Input Images', visible = False))
        outputs.append(gr.Tab(label = 'Color Input Images', visible = False))
        outputs.append(gr.Tab(label = 'Plant Analysis', visible = False))
        outputs.append(gr.Tab(label = 'Plant Statistics', visible = False))
        outputs.append(gr.Gallery(label = 'Uploaded Raw Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = False))
        outputs.append(gr.Gallery(label = 'Color Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = False))
        outputs.append(gr.Gallery(label = 'Plant Analysis',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = False))
        outputs.append(gr.Textbox(label = 'Estimated Plant Height is ',
                                 show_label = True,
                                 visible = False))
        outputs.append(gr.Button(value = 'CLEAR',
                              visible = False))

        return outputs
