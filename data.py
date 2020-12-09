import pandas as pd

# list of dataset names (to user)
names = ['exam_scores', 'petroleum_consumption', 'study_hours', 'wine_quality']

# user-function which returns a pandas df for a given dataset name
def load_dataframe(file):
    assert file in names, "Filename is not in list of datasets. See documentation for list of acceptable dataset names."
    
    print(f"Creating pandas DataFrame for {file} dataset...")
    
    if file == 'exam_scores':
        df = pd.read_csv('datasets/exam_scores_admission.txt', sep=',', header=None)
        df.columns = ['Exam_1', 'Exam_2', 'Admitted']
        
    elif file == 'petroleum_consumption':
        df = pd.read_csv('datasets/petrol_consumption.csv', sep=',')
        df.columns = ['Petrol_Tax', 'Average_Income', 'Paved_Highways', 'Population_Driver_Licence_Perc', 'Petrol_Consumption']
    
    elif file == 'study_hours':
        df = pd.read_csv('datasets/study_hours_scores.csv', sep=',')
        
    elif file == 'wine_quality':
        df = pd.read_csv('datasets/winequality.csv', sep=',')
        
    return df



