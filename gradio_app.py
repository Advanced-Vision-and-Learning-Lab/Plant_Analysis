from GUI import GUI

import sys

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

with demo:

    demo.load(read_logs, None, None, every=1)

demo.launch(share = False)
read_logs()
