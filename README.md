[![pyJedAI Tests](https://github.com/Nikoletos-K/pyJedAI-Dev/actions/workflows/tests.yml/badge.svg)](https://github.com/Nikoletos-K/pyJedAI-Dev/actions/workflows/tests.yml)

__This repo is the development repo for [pyJedAI](https://github.com/Nikoletos-K/pyJedAI).__

> Development to production process:
> 1. Assure that all tests are passing in dev.
> 2. Pass each feature seperatelly.
> 3. Update and run demo notebooks


<details open>
<summary><b>PyPI upload manual</b>  </summary>

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
<summary>Embeddings with NN approach [ongoing]</summary>

![pyJedAI](https://user-images.githubusercontent.com/47646955/189627063-8536a4fd-cc0e-45ec-a038-cff1a3746570.jpg)

</details>

<details>
<summary>Reading Process</summary>

![pyJedAI](https://user-images.githubusercontent.com/47646955/190148478-2221e67c-b694-4116-aa64-3d6a6a88be7e.jpg)

</details>

