import pandas as pd
from pathlib import Path
from clearml import Task, Dataset, StorageManager

task = Task.init(project_name='ClearML examples/Urbansounds',
                 task_name='download data')

configuration = {
    'selected_classes': ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                         'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
}
task.connect(configuration)


def get_urbansound8k():
    # Download UrbanSound8K dataset (https://urbansounddataset.weebly.com/urbansound8k.html)
    # For simplicity we will use here a subset of that dataset using clearml StorageManager
    path_to_urbansound8k = StorageManager.get_local_copy(
        "https://allegro-datasets.s3.amazonaws.com/clearml/UrbanSound8K.zip",
        extract_archive=True)
    path_to_urbansound8k_csv = Path(path_to_urbansound8k) / 'UrbanSound8K' / 'metadata' / 'UrbanSound8K.csv'
    path_to_urbansound8k_audio = Path(path_to_urbansound8k) / 'UrbanSound8K' / 'audio'

    return path_to_urbansound8k_csv, path_to_urbansound8k_audio


def log_dataset_statistics(dataset_task, metadata):
    histogram_data = metadata['class'].value_counts()
    dataset_task.get_logger().report_table(
        title='Raw Dataset Metadata',
        series='Raw Dataset Metadata',
        table_plot=metadata
    )
    dataset_task.get_logger().report_histogram(
        title='Class distribution',
        series='Class distribution',
        values=histogram_data,
        iteration=0,
        xlabels=histogram_data.index.tolist(),
        yaxis='Amount of samples'
    )


def build_clearml_dataset():
    # Get a local copy of both the data and the labels
    path_to_urbansound8k_csv, path_to_urbansound8k_audio = get_urbansound8k()
    urbansound8k_metadata = pd.read_csv(path_to_urbansound8k_csv)
    # Subset the data to only include the classes we want
    urbansound8k_metadata = \
        urbansound8k_metadata[urbansound8k_metadata['class'].isin(configuration['selected_classes'])]

    # TODO: Create train and test folders based on labels and folds
    # for each sample, we check in which fold it is and which label it has
    # then we move it there?

    # Create a pandas dataframe containing labels and other info we need later (fold is for train test split)
    metadata = pd.DataFrame({
        'fold': urbansound8k_metadata.loc[:, 'fold'],
        'filepath': ('fold' + urbansound8k_metadata.loc[:, 'fold'].astype(str)
                     + '/' + urbansound8k_metadata.loc[:, 'slice_file_name'].astype(str)),
        'label': urbansound8k_metadata.loc[:, 'classID']
    })

    # Now create a clearml dataset to start versioning our changes and make it much easier to get the right data
    # in other tasks as well as on different machines
    dataset = Dataset.create(
        dataset_name='UrbanSounds example',
        dataset_project='ClearML examples/Urbansounds'
    )
    # TODO: confusing naming
    # TODO: Add add_metadata to SDK as a wrapperish for upload_artifact
    # A dataset is a task like any other, so we can add plots and artifacts etc. to it. But to do that we need to get
    # the underlying task object first.
    dataset_task = Task.get_task(task_id=dataset.id)
    # Add the local files we downloaded earlier
    dataset.add_files(path_to_urbansound8k_audio)
    # Add the metadata in pandas format, we can now see it in the webUI and have it be easily accessible
    # TODO: change this to dataset.upload_metadata, only sectioning will have a prefix on the artifact
    # Also need get_metadata which will work with this prefix
    dataset_task.upload_artifact(name='metadata', artifact_object=metadata)
    # Finalize and upload the data and labels of the dataset
    dataset.finalize(auto_upload=True)
    # Make sure we wait until everything is uploaded
    dataset_task.flush(wait_for_uploads=True)
    # Let's add some cool graphs as statistics in the plots section!
    log_dataset_statistics(dataset_task, urbansound8k_metadata)


if __name__ == '__main__':
    build_clearml_dataset()
