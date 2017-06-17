#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 18:33:47 2017

@author: ricorodriguez

Analyzing the human connectome with respect to fMRI data.
"""
import pandas

wm_dataset = pandas.read_csv('datasets/WU-Minn_HCP_data.csv')

wm_stats = wm_dataset.describe()

