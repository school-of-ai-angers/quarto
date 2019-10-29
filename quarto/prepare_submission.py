# You can run this file with `python -m quarto.prepare_submission` to check your submission and prepare the ZIP file

from quarto.environment import Environment
from quarto.train import run_duel
from zipfile import ZipFile, ZIP_DEFLATED
import os
import shutil

# In the arena, the cwd is the base of the zip
os.chdir('quarto/submission')

# Load module
print('Load your module... ', end='')
try:
    from quarto.submission.player import Player
except:
    raise Exception(
        'Failed to import your Python module. You may want to re-check your code in quarto/player.py')
print('OK')

# Play against self
env = Environment()
print('Test it against itself... ', end='')
try:
    p1 = Player(False)
    p2 = Player(False)
    run_duel(env, p1, p2, 1000)
except:
    raise Exception(
        'You player failed when playing against itself. You may want to re-check your logic')
print('OK')

# Prepare zip file
# NB: write it to the parent to avoid a nasty loop
print('Create zip... ', end='')
try:
    os.remove('../submission.zip')
except FileNotFoundError as e:
    pass
shutil.make_archive('../submission', 'zip')
print('OK')

print('All set! Please submit the file quarto/submission.zip')
