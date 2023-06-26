# Development repo for [pyJedAI](https://github.com/Nikoletos-K/pyJedAI)
Contains all the files of pyjedai. This repo is for experiments. Commit only here.

#### Usefull links
- __PyPI__: [https://pypi.org/project/pyjedai/](https://pypi.org/project/pyjedai/)
- __TestPyPI__: [https://test.pypi.org/project/pyjedai/](https://test.pypi.org/project/pyjedai/)
- __Anaconda__: furute work

## Useful info

<details>
<summary>PyPI upload manual</summary>

1. Move all new files (production repo `/tests/*` and `/pyjedai/*`) to the `/pyJedAI-Dev/pypi/pyjedai/.`
2. Go to `/pyJedAI-Dev/pypi/pyjedai/.` folder and run:
      ```
      py -m build
      twine upload -u Nikoletos-K -p pyjedai2022 -r pypi .\dist\* --config-file ..\.pypirc --verbose
      ```
   where dist is the directory with the files that will be uploaded.
3. If everything is ok, test ```pip install pyjedai```.
4. Pypi token "github-automation"
   ```
   pypi-AgEIcHlwaS5vcmcCJDQxYmFlNTQwLTA2NDgtNDViNi1hZmIxLTM1YmI0YmI1OTM2NgACD1sxLFsicHlqZWRhaSJdXQACLFsyLFsiNTYwNTZkZjctM2QwNS00ZWQ5LWFkOWYtMzE4N2NjYzNjN2IwIl1dAAAGIJfnzGM5PO9O1AkGkKzt7o4Qnt66oTEuNX8k2A47Qb1i
   ```
   
[Link to instructions](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

</details>

<details>
<summary>PyPI upload from public repo </summary>

</details>

<details>
<summary>Conda Virtual Environment</summary>

1. Create env: `conda create --name {env_name} {python==3.7.5}`
2. Activate env: `conda activate {env_name}`
3. Disable env: `conda deactivate`
3. Install all dependencies: `pip install -r requirements.txt`
4. List of packages in current env: `conda list`
5. Delete env: `conda env remove -n env_name`

[Link to instructions](https://www.machinelearningplus.com/deployment/conda-create-environment-and-everything-you-need-to-know-to-manage-conda-virtual-environment/)

</details>

<details>
<summary>Code profiling</summary>

1. Run with profiler: `python -m cProfile _profiling.py`
2. Save stats: `python -m cProfile -o _profiling.stats _profiling.py`
3. View stats: `python -m pstats _profiling.stats`

[Link to instructions](https://machinelearningmastery.com/profiling-python-code/)

</details>

<details>
<summary>Jekyll website dev</summary>

If first time, install Ruby and after run ```gem install jekyll bundler```

For local deployment and testing:

1. ```cd /webpage```
2.  ```bundle exec jekyll serve```, if fails run ```bundle add webrick```
3. Open localhost:4000


</details>

<details>
<summary>Readthedocs website dev</summary>
Link: https://pyjedai.rtfd.io

For local deployment and testing:

1. Go to pyJedAI public repo
3. Run ```jupyter-book build docs/ ```
4. Open index.html to a browser

</details>

<details>
<summary>View results with optuna dashboard</summary>

1. Go to pyJedAI public repo
2. ```pip install optuna-dashboard ```
3. ```optuna-dashboard sqlite:///pyjedai.db``` , at the dir containing pyjedai.db file

</details>




