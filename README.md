[![pyJedAI Tests](https://github.com/Nikoletos-K/pyJedAI-Dev/actions/workflows/tests.yml/badge.svg)](https://github.com/Nikoletos-K/pyJedAI-Dev/actions/workflows/tests.yml)

# Development repo for [pyJedAI](https://github.com/Nikoletos-K/pyJedAI)
Contains all the files of pyjedai. This repo is for experiments. Commit only here.

#### Usefull links
__PyPI__: [https://pypi.org/project/pyjedai/](https://pypi.org/project/pyjedai/)

__Brainstorming-sheet__: [google-sheet](https://docs.google.com/spreadsheets/d/17AseLUaQrdLWbE5gDQI-Lu-JhnqdYO7o0PNG10vzAVg/edit?usp=sharing)


## Development to production process:
1. Assure that all tests are passing in dev.
2. Pass each feature seperatelly.
3. Update and run demo notebooks

## Visualizations

<details>
<summary>Embeddings with NN approach [ongoing]</summary>

![pyJedAI](https://user-images.githubusercontent.com/47646955/189627063-8536a4fd-cc0e-45ec-a038-cff1a3746570.jpg)

</details>

<details>
<summary>Reading Process</summary>

![pyJedAI](https://user-images.githubusercontent.com/47646955/190148478-2221e67c-b694-4116-aa64-3d6a6a88be7e.jpg)

</details>

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

[Link to instructions](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

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
<summary>Optimization ideas</summary>

1. Avoid pd.itterows()
2. Avoid looping in a set -> transform it first with list()
3. PyPy      

[Link to instructions](https://machinelearningmastery.com/profiling-python-code/)

</details>


