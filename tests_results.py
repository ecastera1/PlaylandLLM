#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:52:23 2023
@author: Enrique Castera
"""

import pandas as pd

my_columns = [
    'model',
    'device',
    'total_time',
    'avg_chars_per_sec',
    'avg_response_length']

df = pd.DataFrame(columns=my_columns)

def add_test_result(
        model,
        device,
        total_time,
        avg_chars_per_sec,
        avg_response_length):
    new_row = [
        str(model),
        str(device),
        str(total_time),
        str(avg_chars_per_sec),
        str(avg_response_length)
    ]

    print(f"add_test_result: {new_row} length {df.shape[0]}")
    df.loc[df.shape[0]] = new_row

    return


def save_results(CSV_FILE, PICKLE_FILE):
    df.to_csv(CSV_FILE, sep=',', quotechar='"')
    print("save_results: Written " + CSV_FILE)
    df.to_pickle(PICKLE_FILE)  # where to save it, usually as a .pkl
    print("save_results: Written " + PICKLE_FILE)
