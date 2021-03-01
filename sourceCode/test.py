import pandas as pd

def flood_classifier(filename, fd, validating=0, braek=None):
    data1 = pd.read_excel('/home/boyking/project /Rainfall_Flood/sourceCode/data/' + filename + '.xlsx')
    data1.head
    print("data1")
    
