import requests
import time
import random
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='Run inference data to serving.')
parser.add_argument('--drift', action='store_true', help='Send drifted data instead of normal data.')
args = parser.parse_args()

data_path = Path('data/')

synthetic_df = pd.read_csv(data_path / 'synthetic.csv')
synthetic_df_drift = pd.read_csv(data_path / 'synthetic_drift.csv')

if args.drift:
    print('Sending drifted data')
    df = synthetic_df_drift
else:
    print("Sending normal data")
    df = synthetic_df

while True:
    response = requests.post(
        url='http://127.0.0.1:8080/serve/asteroid',
        headers={'accept': 'application/json', 'Content-Type': 'application/json'},
        json=df.loc[random.randint(0, len(df) - 1), :].to_dict()
    )
    if response.status_code != 200:
        print(f"Bad request! {response.content}")
    time.sleep(random.randrange(0, 1))