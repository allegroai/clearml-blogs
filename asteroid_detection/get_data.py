from clearml import Task, Dataset

import global_config
from data import database


task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='get data',
    task_type='data_processing',
    reuse_last_task_id=False
)

config = {
    'query_date': '2022-01-01'
}
task.connect(config)


# Get the data and a path to the file
query = 'SELECT * FROM asteroids WHERE strftime("%Y-%m-%d", `date`) <= strftime("%Y-%m-%d", "{}")'.format(config['query_date'])
df, data_path = database.query_database_to_df(query=query)
print(f"Dataset downloaded to: {data_path}")
print(df.head())

# Create a ClearML dataset
dataset = Dataset.create(
    dataset_name='raw_asteroid_dataset',
    dataset_project=global_config.PROJECT_NAME
)
# Add the local files we downloaded earlier
dataset.add_files(data_path)
# Let's add some cool graphs as statistics in the plots section!
dataset.get_logger().report_table(title='Asteroid Data', series='head', table_plot=df.head())
# Finalize and upload the data and labels of the dataset
dataset.finalize(auto_upload=True)

print(f"Created dataset with ID: {dataset.id}")
print(f"Data size: {len(df)}")
