from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from clearml import Dataset, Task
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

# Connecting ClearML with the current process,
# from here on everything is logged automatically
import global_config

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='model training',
    output_uri=True
)

# Set default docker
task.set_base_docker(docker_image="python:3.7")

# Training args
training_args = {
    'eval_metric': "rmse",
    'objective': 'reg:squarederror',
    'test_size': 0.2,
    'random_state': 42,
    'num_boost_round': 100
}
task.connect(training_args)

# Load our Dataset
local_path = Dataset.get(
    dataset_name='preprocessed_asteroid_dataset',
    dataset_project=global_config.PROJECT_NAME
).get_local_copy()
local_path = Path(local_path)
# local_path = Path('data/preprocessed_data')
X = pd.read_csv(local_path / 'X.csv', index_col=0)
y = pd.read_csv(local_path / 'y.csv', index_col=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training_args['test_size'], random_state=training_args['random_state'])
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train
bst = xgb.train(
    training_args,
    dtrain,
    num_boost_round=training_args['num_boost_round'],
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0
)

bst.save_model("best_model")
plot_importance(bst)
plt.show()

preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
accuracy = accuracy_score(y_test['Hazardous'].to_list(), predictions)
recall = recall_score(y_test['Hazardous'].to_list(), predictions)
print(f"Model trained with accuracy: {accuracy} and recall: {recall}")
# Save the actual accuracy as an artifact so we can get it as part of the pipeline
task.get_logger().report_scalar(
    title='Performance',
    series='Accuracy',
    value=accuracy,
    iteration=0
)
task.get_logger().report_scalar(
    title='Performance',
    series='Recall',
    value=recall,
    iteration=0
)
print("Done")
