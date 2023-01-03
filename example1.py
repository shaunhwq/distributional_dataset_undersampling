#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:44:36 2019
@author: bbonik

Example script to demonstrate the use of the distributional undersampling
technique. A 6-dimensional dataset is loaded. Then the undersampling function 
is called, in order to create a balanced subset across all 6 dimensions. 
Different tarket distributions can be achieved by using the correct input
string.
"""
import pandas as pd
from distributional_undersampling import undersample_dataset


if __name__ == '__main__':
    input_csv = "test.csv"
    columns_to_undersample = ["test_col1", "test_col2", "test_col3"]
    output_csv = "test_undersampled.csv"

    df = pd.read_csv(input_csv)
    data = df[columns_to_undersample].to_numpy()

    indices_to_keep = undersample_dataset(data=data,
                                          data_to_keep=30,
                                          target_distribution="uniform",
                                          data_scaling="minmax",
                                          bins=10,
                                          lamda=0.5,
                                          verbose=False,
                                          scatterplot_matrix=False)

    df.loc[indices_to_keep].to_csv(output_csv)
