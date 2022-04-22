#!/bin/bash
PYTHON=/Users/chris/miniconda3/envs/coronavirus/bin/python
cd /Users/chris/coronavirus/
$PYTHON download.py daily
$PYTHON msoa_download.py
$PYTHON msoa_check.py --composite
