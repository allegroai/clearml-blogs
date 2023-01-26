"""This is a mock module and should be replaced with your actual database connector."""
from pathlib import Path
import pandas as pd
from pandasql import sqldf
from datetime import datetime, timedelta


def query_database_to_df(query='SELECT * FROM asteroids'):
    # Get the data as CSV
    data_path = Path('data/nasa.csv')
    out_path = Path('/tmp/nasa.csv')

    # Create a dataframe as mock for the database
    asteroids = pd.read_csv(data_path)

    # Add some mock dates
    asteroids['date'] = [datetime.now() - i*timedelta(days=1) for i in range(len(asteroids))]

    # Query the df base on the argument
    asteroids = sqldf(query, locals())

    # Save resulting DF to disk so it can be added to a clearml dataset as a file
    asteroids.to_csv(out_path)

    return asteroids, out_path