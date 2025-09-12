# Welcome to my implementation of Beam Search

This repository became a mess during my project, but I tried to clean it for the next person using it.


The _EMM.py_ file is heavily flawed. It was taken from a previous project and over time I found and had to fix more and more about it, which is only implemented in _EMM_fixed.py_
Some things are just flaws with the original Beam search idea, but some flaws are just that it was implemented incorrectly. I kept it around for if you wanted to look at legacy.

_subgroup.py, beam.py, description.py, helper_functions.py and evaluation_metrics.py_ are all helper functions or parts of the EMM file. All of these have been adapted to work well with EMM_fixed.
Some of these don't work with Jaccard Beam search, and all the relevant helper functions are in _Jaccard.py_ for that.

_Generation.py_ is used to generate specific types of sets of data that I wanted to run experiments on. I am sure that the skeleton of this code can be useful for future projects that generate data, but you probably want to come up with your own idea of what kind of structure the data should have.

When looking at how the beam search algorithm should be ran, I advise you to look at _experiment.py_.

For the experiments and results folder, some import statements might have broken as I moved them into folders to be clearer when looking at the repository.

I tried to keep things decently commented while working, but undoubtedly you will have questions that I will not answer. 


**P.S.**
There are a few files that I haven't touched in so long and might have become unused. I am talking about visualization.py, workers.py, util.py, dissassembly.ipynb (and anything that has "first commit" as its last commit)
I don't know if anything relies on these files so I haven't removed them.

Lastly, pattern_team.py is a pattern team that tries to use all the created models to create a joined model. I wrote a very short bit about this in my appendix.

