# GUITAR CLASSIFIER-BASED ANNOTATOR
<p align="center">
  <img src="logos/Logo_TSP_IP_Paris_-_Baseline_Noir.png" width="30%" style="vertical-align: middle;"/>
  <img src="logos/cassiopee-logo.png" width="30%" style="vertical-align: middle;"/>
</p>

## GENERAL DESCRIPTION
A poject made for school, in the Cassiopée project of Télécom Sudparis.
## REQUIREMENTS
 * Requires python 3.9
 * For python dependencies, please run :
   ```pip install requirements.txt```

## FUNCTIONALITIES

 * codebase can be run by using `python3.9 codebase/__init__.py`.
   * Currently, is used only to test the jams reading capabilities using code found on [the official github page of the database](https://github.com/marl/GuitarSet).
 * classifier (WIP) can be run using `python3.9 codebase/classify.py`. This code is meant to be integrated within the final product. As of now, reduces the entire database to a numpy matrix.