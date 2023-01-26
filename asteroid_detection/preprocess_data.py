import os.path
from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

import global_config

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='preprocess data',
    task_type='data_processing',
    reuse_last_task_id=False
)

# Create the folder we'll output the preprocessed data into
preprocessed_data_folder = Path('/tmp')
if not os.path.exists(preprocessed_data_folder):
    os.makedirs(preprocessed_data_folder)

# Get the dataset
dataset = Dataset.get(
    dataset_project=global_config.PROJECT_NAME,
    dataset_name='raw_asteroid_dataset',
)
local_folder = dataset.get_local_copy()
print(f"Using dataset ID: {dataset.id}")

# Clean up the data a little bit
df = pd.read_csv((Path(local_folder) / 'nasa.csv'))
df['avg_dia'] = df[['Est Dia in KM(min)', 'Est Dia in KM(max)']].mean(axis=1)
X = df[['Absolute Magnitude', 'avg_dia', 'Relative Velocity km per hr', 'Miss Dist.(kilometers)', 'Orbit Uncertainity',
        'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis',
        'Inclination', 'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
        'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']]
X.to_csv(path_or_buf=preprocessed_data_folder / 'X.csv')
print(f"Preprocessed data X")
print(X.head())

y = pd.DataFrame(df['Hazardous'].astype(int))
y.to_csv(path_or_buf=preprocessed_data_folder / 'y.csv')
print(f"Preprocessed data y")
print(y.head())

# Create a new version of the dataset, which is cleaned up
new_dataset = Dataset.create(
    dataset_project=dataset.project,
    dataset_name='preprocessed_asteroid_dataset',
    parent_datasets=[dataset]
)
new_dataset.add_files(preprocessed_data_folder / 'X.csv')
new_dataset.add_files(preprocessed_data_folder / 'y.csv')
new_dataset.get_logger().report_table(title='X data', series='head', table_plot=X.head())
new_dataset.get_logger().report_table(title='y data', series='head', table_plot=y.head())
new_dataset.finalize(auto_upload=True)

# Log to console which dataset ID was created
print(f"Created preprocessed dataset with ID: {new_dataset.id}")
