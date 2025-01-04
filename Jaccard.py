import sys
import logging
import pandas as pd
import numpy as np
import time

import seaborn
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Any, List, Optional, Union

from util import downsize
from subgroup import Subgroup
from visualization import visualizations
from description import Description
from helper_functions import regression, create_subgroup_lists


def jaccard(set1: set, set2:set):
    """This function takes 2 sets of indexes and returns the jaccard index of the 2 sets"""
    union = len(set1 | set2)
    intersect = len(set1 & set2)
    return intersect/union


def jaccardmatrix(subgroups):
    """This function takes the list of subgroups created by EMM.beam.subgroups
    and returns a matrix of Jaccard indexes"""
    subgroupdataindex = [set([int(i) for i in sg.data.index]) for sg in subgroups]
    n = len(subgroupdataindex)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = jaccard(subgroupdataindex[i], subgroupdataindex[j])
    return matrix


def jaccardmaxlist(subgroups):
    """This function takes the list of subgroups created by EMM.beam.subgroups
    and returns a list of Jaccard indexes"""
    subgroupdataindex = [set([int(i) for i in sg.data.index]) for sg in subgroups] # creates a list of used indexes per subgroup
    n = len(subgroupdataindex)
    lst = []
    for i in range(n):
        fu = []
        for j in range(n):
            if i != j:
                fu.append(jaccard(subgroupdataindex[i], subgroupdataindex[j]))
        lst.append(max(fu)) # if this returns an error, likely your candidate size = 1, make it at least 2 (preferably more)
    return lst


def jaccardmax(candidate, subgroups):
    """This function takes a candidate subgroup and the currently existing subgroups of a Jaccard_Beam
    and returns a matrix of Jaccard indexes"""
    subgroupdataindex = [set([int(i) for i in sg.data.index]) for sg in subgroups]
    candidateindex = set([int(i) for i in candidate.data.index])
    n = len(subgroupdataindex)
    lst = []
    for i in range(n):
        lst.append(jaccard(candidateindex, subgroupdataindex[i]))
    return max(lst)


class Jaccard_EMM:
    def __init__(self, width: int, depth: int=1, 
                 strategy: str = 'maximize', n_bins: int = 10, bin_strategy: Optional[str] = 'equidepth',
                 candidate_size: int = None, log_level=50):
        logging.basicConfig(filename=None, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        if None:  ### THIS IS OLD STUFF THAT WAS USED TO MULTITHREAD AND USE OTHER EVALUATION METRICS, WE DONT USE IT
            # self.evaluation_metric = evaluation_metric  
            # if n_jobs == -1:
            #     self.n_jobs = cpu_count()
            # else:
            #     self.n_jobs = min(n_jobs, cpu_count())
            # if hasattr(evaluation_metric, '__call__'):
            #     self.evaluation_function = evaluation_metric
            # else:
            #     try:
            #         self.evaluation_function = metrics[evaluation_metric]
            #     except KeyError:
            #         raise ValueError(f"Nu such metric: {evaluation_metric}")
            pass
        self.settings = dict(
            strategy=strategy,
            width=width,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            candidate_size=candidate_size
        )
        self.beam = None
        self.target_columns = None
        self.dataset_target = None
        self.dataset = None
        self.depth = depth

    def set_data(self, data: pd.DataFrame, target_cols: Union[List[str], str], descriptive_cols: List[str] = None):
        """This method takes a dataset and prepares it for the beam search
        target cols is supposed to be a list of 2 strings, representing column names"""
        logging.info("Start")
        data, translations = downsize(deepcopy(data))
        self.settings['object_cols'] = translations
        dataset = Subgroup(data, Description('all'))
        _, dataset.target = regression(data[target_cols], data[target_cols],[0])
        self.regressioncache = [dataset.target]
        self.beam = Jaccard_Beam(dataset, self.settings)
        self.jaccard_matrix = None
        target_cols = list(target_cols, )
        if descriptive_cols is None:
            self.descriptive_cols = [c for c in data.columns if c not in target_cols]
        elif any(c in descriptive_cols for c in target_cols):
            raise ValueError("The target and descriptive columns may not overlap!")
        else:
            self.descriptive_cols = descriptive_cols
        if any(c not in data.columns for c in self.descriptive_cols + target_cols):
            raise ValueError("All specified columns should be present in the dataset")
        self.dataset_target = data[target_cols]
        self.target_columns = target_cols

    def subgroupify(self):
        """This method creates all possible subgroups in the current state"""
        subgroups = []
        for subgroup in self.beam.subgroups:
                for col in self.descriptive_cols:
                    newgroups = create_subgroup_lists(subgroup, col, self.settings)
                    subgroups = subgroups + newgroups
        self.candidates = subgroups
    
    def calc_score(self, print_result = True):
        """This method calculates scores for all candidate subgroups made in the subgroupify method"""
        for candidate in self.candidates:
            candidate_target = candidate.data[self.target_columns]
            candidate.score, candidate.target = regression(candidate_target, self.dataset_target, comparecache=self.regressioncache)
            self.beam.add(candidate) # the jacscore is calculated when adding to the beam
        self.beam.select_cover_based()
        # update regressioncache after selecting subgroups
        for subgroup in self.beam.subgroups:
            self.regressioncache.append(subgroup.target)
        if print_result == True:
            self.beam.print()
        else:
            logging.info("finished an iteration")
    
    def increase_depth(self,iterations = 1, print_result_between_iterations=False):
        for _ in range(iterations):
            self.subgroupify()
            self.calc_score(print_result=print_result_between_iterations)
        if print_result_between_iterations == False:
            self.beam.print()
    
    def search(self, data, target_cols):
        self.set_data(data, target_cols)
        self.increase_depth(self.depth)



class Jaccard_Beam:
    def __init__(self, subgroup: Subgroup, settings: dict):
        self.subgroups = [subgroup]
        self.candidates = []
        self.items = 1
        self.max_items = settings['width']
        try:
            self.candidate_size = int(settings['candidate_size'])
        except (KeyError, TypeError):  # if no candidate size is given, make it width
            if settings['width'] > 1:
                self.candidate_size = settings['width']
            else: # if width is 1, make candidate size 2, (to avoid errors)
                self.candidate_size = 2
        self.strategy = settings['strategy']
        self.min_score = None
        self.scores = []
        self.jaclist = []
        self.jacscores = []
        self.full = False

    def add(self, subgroup: Subgroup):
        """Adds a subgroup to the beam
        Before adding, checks if the beam isnt full
        If the beam is full, checks if this subgroup improves the beam"""
        if self.full:
            jacscore = subgroup.score * ( 1 -jaccardmax(subgroup, self.candidates))
            if ((self.strategy == 'maximize' and jacscore > self.min_score) or \
                    (self.strategy == 'minimize' and jacscore < self.min_score)):
                idx = self.jacscores.index(self.min_score)
                del self.scores[idx]
                del self.candidates[idx]
                del self.jacscores[idx]
                self.candidates.append(subgroup)
                self.update_jacscores() # all scores need to be updates, now that we deleted another subgroup and addedd a new one
                self.min_score = min(self.jacscores) if self.strategy == 'maximize' else max(self.jacscores)
        elif len(self.candidates) < self.candidate_size:
            self.candidates.append(subgroup)
        else: # The list is full, so we are going to apply jaccard index scoring now
            self.full = True
            self.update_jacscores()
            self.add(subgroup)

    def update_jacscores(self):
        """This updates all the scores by multiplying jaccard index with the score"""
        self.jaclist = jaccardmaxlist(self.candidates)
        self.scores = [subgroup.score for subgroup in self.candidates]
        self.jacscores = [self.scores[i] * (1 - self.jaclist[i]) for i in range(len(self.scores))]
        self.min_score = min(self.jacscores) if self.strategy == 'maximize' else max(self.jacscores)
        for i, candidate in enumerate(self.candidates): ## this is now done in select cover based
            candidate.jacscore = self.jacscores[i]


    def sort(self, attribute: str = 'score') -> None:
        """Sort the candidates on score"""
        if attribute == 'score':
            self.candidates.sort(key=lambda x: x.jacscore, reverse=(self.strategy == 'maximize'))
        #             self.subgroups.sort(key=lambda x: x.jacscore, reverse=(self.strategy == 'maximize'))
        elif attribute == 'coverage':  # not working with jacscore, this is basically deprecated
            self.candidates.sort(
                key=lambda x: x.score * (x.coverage if (self.strategy == 'maximize') else (1 - x.coverage)),
                reverse=(self.strategy == 'maximize'))
            print("WARNING!!! you are using a deprecated function")
        else:
            raise ValueError("Invalid sort attribute")

    def select_cover_based(self):
        """This selects the top sets to print/return at the end of the search on a depth"""
        self.update_jacscores()
        self.sort()
        self.update_jacscores() # need to do this to make sure the jaccard list is in the same order as the subgroups
        self.full = False
        if self.candidate_size > self.max_items: # if the subgroup is too big,
            index = np.array([])
            for subgroup in self.candidates:
                subgroup.coverage = 1 - \
                            (np.intersect1d(subgroup.data.index.values, index).size / subgroup.data.index.size)
                index = np.unique(np.concatenate((index, subgroup.data.index.values)))
            self.sort(attribute='score')
        self.subgroups = self.candidates[:self.max_items]
        self.scores = [s.score for s in self.subgroups]
        self.min_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)
        for i, subgroup in enumerate(self.subgroups):
            subgroup.jaccard = self.jaclist[i]
            subgroup.jacscore = self.jacscores[i]
        self.candidates = []
        # self.jaclist = []
        # self.jacscores = []

    def decrypt_descriptions(self, translation):
        for s in self.subgroups:
            s.decrypt_description(translation)

    def print(self):
        self.sort() # this used to be on coverage, but this is deprecated; it doesn't work in current version
        logging.debug("-" * 20)
        for s in self.subgroups:
            s.printreal()

# if __name__ =="__main__":
#     df = pd.read_csv('example/data/german-credit-scoring.csv', sep=";")
#     target_columns = ['Duration in months', 'Credit amount']
#     w = 100
#     d = 4
#     # jaccardbEaMM = Jaccard_EMM(width=w, depth=d, evaluation_metric='regression', n_jobs=-1, log_level=2)
#     # jaccardbEaMM.search(df, target_cols=target_columns)
#     # matrix = jaccardmatrix(jaccardbEaMM.beam.subgroups)
#     # seaborn.heatmap(matrix)
#     # plt.savefig(f'width{w},depth{d}jaccardbeam-Jaccardheatmap.png')
#     # plt.clf()

#     w = 100
#     d = 4
#     bEaMM = EMM(width=w, depth=d, evaluation_metric='regression', n_jobs=-1, log_level=2)
#     bEaMM.search(df, target_cols=target_columns)
#     matrix = jaccardmatrix(bEaMM.beam.subgroups)
#     seaborn.heatmap(matrix)
#     plt.savefig(f'width{w},depth{d}normalbeam-Jaccardheatmap.png')



#     # jaccardbEaMM.visualise()



