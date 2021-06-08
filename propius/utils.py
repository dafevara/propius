import os
import re
import csv
import pandas as pd


def stream_local_csv(file_path, options):
    reader = pd.read_csv(file_path, chunksize=1000, **options)
    if reader:
        return reader


def stream_csv_from_s3(s3_file_url, options):
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    reader = pd.read_csv(
        s3_file_url,
        storage_options={
            "key": AWS_ACCESS_KEY_ID,
            "secret": AWS_SECRET_ACCESS_KEY,
        },
        chunksize=1000
    )
    if reader:
        return reader


def stream_csv(file_path, options={}):
    if not file_path:
        raise Exception('Not csv file path found')

    if re.match(r's3://', file_path):
        return stream_csv_from_s3(file_path, options)
    else:
        return stream_local_csv(file_path, options)
