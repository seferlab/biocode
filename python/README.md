BioCode(Python version)
========

A non-parametric framework for learning how biological networks grow

You need to have Python3 and install requirements.txt.
	pip install -r requirements.txt

Running
-------

Sample run for all following 3 networks:
- YeastPPI.edg
- BiologicalCollaboration.edg
- GeneRegulatory.edg

	python runexample.py
	
The runexample.py scripts shows how to run a basic learning and program
selection procedure using Python. The learning procedure is
associated with a variety of genetic algorithm parameters that can be
customized in each example's '.params' file.  Please see the ECJ and
DEAP documentation for more information on what these parameters mean.

To Do
-----
* Finish tests for Python version
