# FineBlade-dataset
Experimental DoS dataset

## System
The dataset has been collected by running a set of synthetic attacks over the <a href ="https://github.com/dedos-project/DeDOS" >DeDoS</a> platform. Attacks are described in the attack file. This specific data requires MySQL.


## Dependencies
```console
# apt-get install libmariadbclient-dev
```
If using pip:
```console
# pip install numpy scipy pandas pyyaml sqlalchemy mysql-python jupyter
```
If using conda:
```console
# conda install numpy scipy pandas pyyaml sqlalchemy mysql-python jupyter
```

## Setup the database
Get the database at https://drive.google.com/open?id=1dek34WgmDwzC8D27_Ijt1Lq7F7qh9yy7 and inject it into a MariaDB or MySQL database previously created.

## Notebook
Configure the fourth cell with your directory and database information. The seventh cell display the attack events,
while the eigth plot the 100th percentile of the feature gathered during the experiment.
