#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 18:33:47 2017

@author: ricorodriguez

Analyzing the human connectome with respect to fMRI data. All data has been
retrieved from the Human Connectome project, the database for which can be
found at:
    https://db.humanconnectome.org/

** DATASETS **

- hcp_data:
    This HCP data release includes high-resolution 3T MR scans from young
    healthy adult twins and non-twin siblings (ages 22-35) using four imaging
    modalities: structural images (T1w and T2w), resting-state fMRI (rfMRI),
    task-fMRI (tfMRI), and high angular resolution diffusion imaging (dMRI).
    Behavioral and other individual subject measure data (both NIH Toolbox and
    non-Toolbox measures) is available on all subjects. MEG data and 7T MR data
    is available for a subset of subjects (twin pairs). The Open Access Dataset
    includes imaging data and most behavioral data. To protect subject privacy,
    some of the data (e.g., which subjects are twins) are part of a Restricted
    Access dataset.

- hcp_retest:
    46 HCP subjects were retested using the full HCP 3T multimodal imaging and
    behavioral protocol. The retest data are released as a separate project to
    fully distinguish it from the first visit. Retest subjects retain the same
    Subject ID numbers as in the 1200 Subjects HCP project. Note: Be sure to
    keep any Retest data separate in your local repository. Downloaded data
    from the Retest project will unpack exactly as from the 1200 Subjects
    project. Be careful not to overwrite a subject’s first visit data with
    their second visit Retest data!

- hcp_lifespan:
    The WU-Minn HCP consortium is acquiring and sharing pilot multimodal
    imaging data acquired across the lifespan, in 6 age groups (4-6, 8-9,
    14-15, 25-35, 45-55, 65-75) and using scanners that differ in field
    strength (3T, 7T) and maximum gradient strength (70-100 mT/m). The scanning
    protocols are similar to those for the WU-Minn Young Adult HCP except
    shorter in duration. The objectives are (i) to enable estimates of effect
    sizes for identifying group differences across the lifespan and (ii) to
    enable comparisons across scanner platforms, including data from the MGH
    Lifespan Pilot. The released data includes unprocessed Phase1a image data
    as collected and with gradient distortion correction applied (indicated by
    "_gdc" in the file names).

- mgh_hcp_adult:
    The MGH HCP team has released diffusion imaging and structural imaging data
    acquired from 35 young adults using the customized MGH Siemens 3T
    Connectome scanner, which has 300 mT/m maximum gradient strength for
    diffusion imaging.
"""
import pandas

# Get data.
hcp_data = pandas.read_csv('datasets/WU-Minn_HCP_data.csv')
hcp_retest = pandas.read_csv('datasets/WU-Minn_HCP_retest.csv')
hcp_lifespan = pandas.read_csv('datasets/WU-Minn_HCP_Lifespan.csv')
mgh_hcp_adult = pandas.read_csv('datasets/MGH_HCP_Adult_Diffusion.csv')

# Get Descriptive statistics.
hcp_stats = hcp_data.describe()
retest_stats = hcp_retest.describe()
lifespan_stats = hcp_lifespan.describe()
mgh_stats = mgh_hcp_adult.describe()

print(hcp_stats, retest_stats, lifespan_stats, mgh_stats)