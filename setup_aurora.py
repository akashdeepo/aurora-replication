"""Fetch the released Aurora model code into vendor/aurora.

The code comes from the authors' repository, github.com/decisionintelligence/Aurora, pinned to commit a247760
(the version tested here). It is not redistributed in this repository. The weights are downloaded separately
from Hugging Face (DecisionIntelligence/Aurora) the first time 02_aurora_forecasts.py runs.

Usage: python setup_aurora.py
"""
import os, shutil, subprocess, tempfile

COMMIT = 'a247760abbc9d17a861bc365c032368d317815f2'
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, 'vendor', 'aurora')

if os.path.isdir(DEST):
    print('vendor/aurora already exists; delete it to fetch again')
else:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['git', 'clone', '--quiet', 'https://github.com/decisionintelligence/Aurora', tmp], check=True)
        subprocess.run(['git', '-C', tmp, 'checkout', '--quiet', COMMIT], check=True)
        shutil.copytree(os.path.join(tmp, 'aurora'), DEST, ignore=shutil.ignore_patterns('__pycache__'))
    print('copied the aurora package at commit', COMMIT[:7], 'to vendor/aurora')
