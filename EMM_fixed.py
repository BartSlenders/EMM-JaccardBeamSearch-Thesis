import logging
from copy import deepcopy
from util import downsize
from description import Description
from subgroup import Subgroup
from beam import Beam
from helper_functions import regression, create_subgroup_lists


# EMM INIT
class EMM():
    """Object for standard beam search
    Works only with regression for now"""
    def __init__(self, width, depth=1, strategy = 'maximize',
                 n_bins = 10, bin_strategy = 'equidepth', candidate_size = None, log_level=1 ) -> None:
        """Initialization for the beam search exceptional model mining procedure"""
        logging.basicConfig(filename=None, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
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
    
    def set_data(self, data, target_cols, descriptive_cols = None):
        """This method takes a dataset and prepares it for the beam search
        target cols is supposed to be a list of 2 strings, representing column names"""
        logging.info("Start")
        data, translations = downsize(deepcopy(data))
        self.settings['object_cols'] = translations
        dataset = Subgroup(deepcopy(data), Description('all'), [])
        _, dataset.target = regression(data[target_cols], data[target_cols],comparecache=[0])
        # self.regressioncache = dataset.target
        self.beam = Beam(dataset, self.settings)
        target_cols = list(target_cols,)
        if descriptive_cols == None:
            self.descriptive_cols = [c for c in data.columns if c not in target_cols]
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
                regressioncacheforthissubgroup = deepcopy(subgroup.target)
                subgroup.append_regression_cache(regressioncacheforthissubgroup)
                for col in self.descriptive_cols:
                    newgroups = create_subgroup_lists(subgroup, col, self.settings)
                    subgroups = subgroups + newgroups
        self.candidates = subgroups
    
    def calc_score(self, print_result = True):
        """This method calculates scores for all candidate subgroups made in the subgroupify method"""
        for candidate in self.candidates:
            candidate_target = candidate.data[self.target_columns]
            candidate.score, candidate.target = regression(candidate_target, self.dataset_target, comparecache=candidate.regressioncache)
            self.beam.add(candidate)
        self.beam.select_cover_based()
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
