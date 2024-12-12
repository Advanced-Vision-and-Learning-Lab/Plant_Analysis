
from GUI import GUI
import cProfile
#import gradio as gr
import sys
import logging
# Suppress Gradio's version warning
logging.getLogger("gradio").setLevel(logging.ERROR)
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

    #sys.stdout.write('Gradio Application\n')
    sys.stdout.flush()
    #with open(log_filename, "r") as f:
        #return f.read()

# launch GUI
gui = GUI()
demo = gui.demo
demo.title="Plant Analysis and Feature Extraction"
with demo:
    # Inject HTML to change the tab title
    #gr.HTML("<script>document.title = 'Plant Phenotyping';</script>")
    demo.load(read_logs, None, None, every=1)
    #demo.load(lambda: '<script>document.title = "My Project Title";</script>', None, None)

#demo.launch(share = False)
# Launch the app with 'inbrowser=True' to open automatically
demo.launch(share=False, inbrowser=True)
read_logs()
