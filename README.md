# GUITAR CLASSIFIER-BASED ANNOTATOR
<p align="center">
  <img src="logos/Logo_TSP_IP_Paris_-_Baseline_Noir.png" width="30%" style="vertical-align: middle;"/>
  <img src="logos/cassiopee-logo.png" width="30%" style="vertical-align: middle;"/>
</p>

## GENERAL DESCRIPTION
A poject made for school, in the Cassiopée project of Télécom Sudparis.
## REQUIREMENTS
 * Requires python 3.10
 * For python dependencies, please run :
   ```pip install requirements.txt```

## Make functionalities

If you can run make files, then you can do most of the work bby just calling the `make` script.
It should generate a brand new model and train it according to the databases available.


To avoid having it compute for 20 hours, please ask for a pre-computed xy folder. It is too heavy to store on github, but should make the script much faster.

## PYTHON FUNCTIONALITIES

 * main application can be run by using `bash app.sh`
 * to train the model, you can use `python3.10 codebase/musicHandler.py` 
 * a display of general functionalities of librosa can be run by using :
   * `python3.10 codebase/__init__.py`
   * `python3.10 codebase/librosaTest.py`
 * Currently, is used only to test the jams reading capabilities using code found on [the official github page of the database](https://github.com/marl/GuitarSet).
 * classifier (WIP) can be run using `python3.9 codebase/classify.py`. This code is meant to be integrated within the final product. As of now, reduces the entire database to a numpy matrix.