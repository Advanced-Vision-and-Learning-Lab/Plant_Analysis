# Plant_Analysis
Plant analysis pipeline for greenhouse data analysis


## Installation
* This repository is tested with Python 3.10. It is recommended to use the same Python version for smoother installation and running.
* Use pip to install the dependencies using requirements.txt file

## Dataset
Download the following dataset zip files from google drive [folder](https://drive.google.com/drive/folders/1qohla3xY_66ueb7C19XPDO1uWk3ZGh2U?usp=sharing) 
* Raw_Images.zip (Contains 7 cotton plants)
* Raw_Images_Duplicated.zip (Duplicated cotton plants to make a dummy dataset of 100 plants. This can be useful to check the latency of the pipeline for a large number of plants)
* Rice_Corn_raw_images_plantwise.zip (Contains Rice and Corn plants)
Unzip these files into a dataset folder and specify the folder path of a dataset you would like as input folder path in the GUI


## Launching GUI
* Run ''' python gradio_app.py ''' to launch the GUI
* Check the GUI usage guide [here](https://plant-analysis-avll.readthedocs.io/en/latest/)
* Use ''' share=True ''' option in ''' gradio_app.py ''' in order to get a publicly shareable link

## Launching Visualization Tool
* Run ''' python gradio_viz_app.py ''' to launch visualization tool
* In the GUI, insert the appropriate results folder path to check the visualization of results

## Code Explanation
* Check comments before each function and variable in the Python files
