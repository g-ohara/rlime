"""Base anchor functions"""
from __future__ import print_function
import numpy as np
import copy
import collections

from anchor import anchor_base
from river import compose
from river import linear_model
from river import preprocessing 

from typing import Callable
from typing import Any 

Predicate = tuple[int, str, int]
Rule = tuple[int, ...]
Mapping = dict[int, Predicate]
Classifier = Callable[[np.ndarray], np.ndarray]
SampleFn = Callable[
        [list[int], int, bool, compose.Pipeline | None, bool], 
        tuple[np.ndarray, np.ndarray, np.ndarray]]
CompleteSampleFn = Callable[[int], float]
State = dict[str, Any]
Anchor = dict[str, Any]

class AnchorBaseBeam(anchor_base.AnchorBaseBeam):
    @staticmethod
    def complete_sample_fn(
            t           : Rule, 
            n           : int, 
            model       : compose.Pipeline,
            sample_fn   : SampleFn,
            state       : State) -> float:
        
        # *****************************************************************
        # Take a sample satisfying the tuple t.
        labels: np.ndarray
        raw_data, data, labels = sample_fn(
                list(t),
                n, 
                True,
                model,
                True)
        # *****************************************************************

        current_idx = state['current_idx']
        # idxs = range(state['data'].shape[0], state['data'].shape[0] + n)
        idxs: range = range(current_idx, current_idx + n)

        if '<U' in str(raw_data.dtype):
            # String types: make sure both string types are of maximum length 
            # to avoid string truncation. E.g., '<U308', '<U290' -> '<U308'
            max_dtype = max(str(state['raw_data'].dtype), str(raw_data.dtype))
            state['raw_data'] = state['raw_data'].astype(max_dtype)
            raw_data = raw_data.astype(max_dtype)

        state['t_idx'][t].update(idxs)
        state['t_nsamples'][t] += n
        state['t_positives'][t] += labels.sum()
        state['data'][idxs] = data
        state['raw_data'][idxs] = raw_data
        if len(labels) > 0:
            state['labels'][idxs] = labels
        state['current_idx'] += n
        if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
            prealloc_size = state['prealloc_size']
            current_idx = data.shape[0]
            state['data'] = np.vstack(
                (state['data'],
                 np.zeros((prealloc_size, data.shape[1]), data.dtype)))
            state['raw_data'] = np.vstack(
                (state['raw_data'],
                 np.zeros((prealloc_size, raw_data.shape[1]),
                          raw_data.dtype)))
            state['labels'] = np.hstack(
                (state['labels'],
                 np.zeros(prealloc_size, labels.dtype)))
        # This can be really slow
        # state['data'] = np.vstack((state['data'], data))
        # state['raw_data'] = np.vstack((state['raw_data'], raw_data))
        # state['labels'] = np.hstack((state['labels'], labels))
        ret: float = labels.sum()
        return ret
    ## end function

    @staticmethod
    def init_surrogate_models(num_models: int) -> list[compose.Pipeline]:
        models: list[compose.Pipeline] = []
        for i in range(num_models):
            preprocessor = preprocessing.StandardScaler()
            model = linear_model.LogisticRegression()
            pipeline = compose.Pipeline(preprocessor, model)
            models.append(pipeline)
        return models
    ## end function

    @staticmethod
    def get_sample_fns(
            sample_fn       : SampleFn,
            tuples          : list[Rule],
            state           : State,
            surrogate_models: list[compose.Pipeline]
            ) -> list[CompleteSampleFn]:
        
        # each sample fn returns number of instances that the surrogate model
        # classified correctly
        sample_fns: list[CompleteSampleFn] = []

        for t, m in zip(tuples, surrogate_models):
            fn = lambda n, t=t, m=m, fn=sample_fn, s=state: AnchorBaseBeam.complete_sample_fn(t, n, m, fn, s)
            sample_fns.append(fn)

        return sample_fns
    ## end function

    @staticmethod
    def generate_cands(
            previous_bests  : list[Rule],
            best_coverage   : float,
            state           : State,
            my_verbose      : bool) -> list[tuple[int, ...]]:

        tuples: list[tuple[int, ...]]
        tuples = AnchorBaseBeam.make_tuples(previous_bests, state)
        tuples = [x for x in tuples
                  if state['t_coverage'][x] > best_coverage]
        return tuples
    ##

    @staticmethod
    def b_best_cands(
            tuples          : list[Rule],
            sample_fn       : SampleFn,
            beam_size       : int,
            epsilon         : float,
            delta           : float,
            batch_size      : int,
            top_n           : int,
            state           : State,
            verbose         : bool,
            verbose_every   : int
            ) -> tuple[
                    list[int],
                    list[tuple[int, ...]],
                    list[CompleteSampleFn],
                    list[compose.Pipeline]]:

        surrogate_models = AnchorBaseBeam.init_surrogate_models(len(tuples)) 
        sample_fns = AnchorBaseBeam.get_sample_fns(
                sample_fn, tuples, state, surrogate_models)
        initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                              state)
        # print tuples, beam_size

        chosen_tuples = AnchorBaseBeam.lucb(
            sample_fns, initial_stats, epsilon, delta, batch_size, top_n,
            verbose, verbose_every)

        # Get candidates from their indexes
        return list(chosen_tuples), [tuples[x] for x in chosen_tuples], sample_fns, surrogate_models
    ##

    @staticmethod
    def update_confidence_bound(
            rule        : tuple[int, ...],
            batch_size  : int,
            beta        : float,
            state       : State) -> tuple[float, float, float]:

        mean = state['t_positives'][rule] / state['t_nsamples'][rule]
        lb = AnchorBaseBeam.dlow_bernoulli(
            mean, beta / state['t_nsamples'][rule])
        ub = AnchorBaseBeam.dup_bernoulli(
            mean, beta / state['t_nsamples'][rule])

        return mean, lb, ub
    ##
            
    @staticmethod
    def largest_valid_cand(
            chosen_tuples       : list[int], 
            tuples              : list[tuple[int, ...]],
            delta               : float,
            beam_size           : int,
            n_features          : int,
            state               : State,
            verbose             : bool,
            desired_confidence  : float,
            epsilon_stop        : float,
            sample_fns          : list[CompleteSampleFn],
            batch_size          : int,
            surrogate_models    : list[compose.Pipeline],
            stop_on_first       : bool,
            ) -> tuple[tuple[int, ...], compose.Pipeline | None, float, bool]:

        best_cand           : Rule 
        best_cand_model     : compose.Pipeline | None
        best_cand_coverage  : float

        best_cand = ()
        best_cand_model = None
        best_cand_coverage = 0.0
        stop_this = False

        # t --- a tuple in best candidates
        # i --- an INDEX to the tuple t
        for i, t in list(zip(chosen_tuples, tuples)):

            # I can choose at most (beam_size - 1) tuples at each step,
            # and there are at most n_feature steps
            beta = np.log(1. / (delta / (1 + (beam_size - 1) * n_features)))

            # Update confidence interval and coverage of the tuple t.
            mean, lb, ub = AnchorBaseBeam.update_confidence_bound(
                    t, batch_size, beta, state)
            coverage = state['t_coverage'][t]
            if verbose:
                print(i, mean, lb, ub)

            # Judge whether the tuple t is an anchor or not.
            while ((mean >= desired_confidence and
                   lb < desired_confidence - epsilon_stop) or
                   (mean < desired_confidence and
                    ub >= desired_confidence + epsilon_stop)):
                sample_fns[i](batch_size)
                mean, lb, ub = AnchorBaseBeam.update_confidence_bound(
                        t, batch_size, beta, state)
            ## end while

            if verbose:
                print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' % (t, mean, lb, ub, coverage, state['t_nsamples'][t]))

            # If the tuple t is the anchor with the provisionally best
            # coverage, update 'best_tuple' and 'best_model'.
            if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                if coverage > best_cand_coverage:
                    best_cand_coverage = coverage
                    best_cand = t
                    best_cand_model = copy.deepcopy(surrogate_models[i])
                    if best_cand_coverage == 1 or stop_on_first:
                        stop_this = True
            ## end if
        ## end for
        return best_cand, best_cand_model, best_cand_coverage, stop_this
    ## 

    @staticmethod
    def anchor_beam(
            sample_fn           : SampleFn,
            delta               : float = 0.05,
            epsilon             : float = 0.1,
            batch_size          : int = 10,
            min_shared_samples  : int = 0,
            desired_confidence  : float = 1.0,
            beam_size           : int = 1,
            verbose             : bool = False,
            epsilon_stop        : float = 0.05,
            min_samples_start   : int = 0,
            max_anchor_size     : int | None = None,
            verbose_every       : int = 1,
            stop_on_first       : bool = False,
            coverage_samples    : int = 10000,
            my_verbose          : bool = False
            ) -> tuple[Anchor, compose.Pipeline | None]:

        anchor: Anchor = {
                'feature': [], 'mean': [], 'precision': [],
                'coverage': [], 'examples': [], 'all_precision': 0}
        _, coverage_data, _ = sample_fn(
                [], coverage_samples, False, None, True)
            

        surrogate_models: list[compose.Pipeline] = AnchorBaseBeam.init_surrogate_models(1) 

        # $B:G>.8D?t$@$1%5%s%W%j%s%0(B
        raw_data, data, labels = sample_fn(
                [], max(1, min_samples_start), True, surrogate_models[0], True)

        # anchor$B$N@:EY$N4|BTCM(B
        if len(labels) > 0:
            mean = np.mean(labels)
        else:
            mean = 0

        beta = np.log(1. / delta)

        # anchor$B$N@:EY$N?.Mj2<8B(B
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])

        while mean > desired_confidence and lb < desired_confidence - epsilon:
            # $B?7$7$/%5%s%W%j%s%0(B
            nraw_data, ndata, nlabels = sample_fn(
                    [], batch_size, True, surrogate_models[0], True)
            # $B?7$7$$%5%s%W%k$r7k9g(B
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            # $B@:EY$NJ?6Q$H?.Mj2<8B$r99?7(B
            mean = labels.mean()
            lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        ##

        # $B?.Mj2<8B$,@)Ls$rK~$?$9$J$i$P=*N;(B
        if lb > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            if my_verbose:
                print('lb (%f) > desired_confidence (%f)' % (lb, desired_confidence))
            return anchor, surrogate_models[0]

        # $B%a%b%j$r3NJ]$7$F%<%mKd$a(B
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]),
                                         data.dtype)))
        raw_data = np.vstack(
            (raw_data, np.zeros((prealloc_size, raw_data.shape[1]),
                                raw_data.dtype)))
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))

        n_features = data.shape[1]
        state: State = {'t_idx': collections.defaultdict(lambda: set()),
                 't_nsamples': collections.defaultdict(lambda: 0.),
                 't_positives': collections.defaultdict(lambda: 0.),
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),
                 't_coverage': collections.defaultdict(lambda: 0.),
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list())
                 }
        current_size = 1


        best_of_size: dict[int, list[Rule]] = {0: []}


        # the rule with the highest coverage of the rules with higher
        # precision than tau in the B best rules
        best_tuple: Rule

        # the surrogate model learned under best_tuple
        best_model: compose.Pipeline | None

        # coverage of best_tuple
        best_coverage: float
        
        best_tuple = ()
        best_model = None
        best_coverage = -1.0

        if max_anchor_size is None:
            max_anchor_size = n_features

        while current_size <= max_anchor_size:

            # newly generated candidate rules
            cands: list[Rule]

            # Call 'GenerateCands' and get new candidate rules.
            cands = AnchorBaseBeam.generate_cands(
                    best_of_size[current_size - 1],
                    best_coverage,
                    state,
                    my_verbose)

            if len(cands) == 0:
                if my_verbose:
                    print('Cannot generate new candidate rules')
                break

            # -----------------------------------------------------------------

            # list of indexes of the B best rules of candidate rules
            chosen_tuples: list[int]

            # list of the B best rules
            # (thus it has the same length as chosen_tuples)
            chosen_rules: list[Rule]

            # list of methods to sample perturbed vectors under the each rule
            # in the candidate rules 
            # (thus it has the same length as cands)
            sample_fns: list[CompleteSampleFn]
            
            # list of surrogate models learned under the each rule in the B
            # best rules
            # (thus it has the same length as cands)
            surrogate_models: list[compose.Pipeline]

            # Call 'B-BestCands' and get the best B candidate rules.
            chosen_tuples, chosen_rules, sample_fns, surrogate_models = AnchorBaseBeam.b_best_cands(
                    cands, sample_fn, beam_size, epsilon, delta, batch_size,
                    min(beam_size, len(cands)), state, verbose, verbose_every)

            best_of_size[current_size] = chosen_rules

            if verbose:
                print('Best of size ', current_size, ':')

            # -----------------------------------------------------------------

            # the rule with the highest coverage of the rules with higher
            # precision than tau in the B best rules
            best_cand: Rule
    
            # the surrogate model learned under best_tuple
            best_cand_model: compose.Pipeline | None
    
            # coverage of best_tuple
            best_cand_coverage: float
        
            # Call 'LargestValidCand' and get the candidate rule with the 
            # highest coverage in the best B candidate rules
            best_cand, best_cand_model, best_cand_coverage, stop_this = AnchorBaseBeam.largest_valid_cand(
                    chosen_tuples, cands, delta, beam_size, n_features, state,
                    verbose, desired_confidence, epsilon_stop, sample_fns,
                    batch_size, surrogate_models, stop_on_first)

            if best_cand_coverage > best_coverage:
                best_tuple = copy.deepcopy(best_cand)
                best_model = copy.deepcopy(best_cand_model)
                best_coverage = best_cand_coverage


            if stop_this:
                if my_verbose:
                    print('Stop This!')
                break
            
            # -----------------------------------------------------------------

            # go to next iteration
            current_size += 1

        ## end while


        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            if verbose:
                print('Could not find an anchor, now doing best of each size')

            # list of the rules with the highest precision in each round
            tuples: list[Rule]
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])

            chosen_tuples, _, sample_fns, surrogate_models = AnchorBaseBeam.b_best_cands(
                    tuples,
                    sample_fn,
                    beam_size,
                    epsilon,
                    delta,
                    batch_size,
                    1,
                    state,
                    verbose,
                    verbose_every)
            best_tuple = tuples[chosen_tuples[0]]
            best_model = surrogate_models[chosen_tuples[0]]
        ## end if

        # return best_tuple, state

        best_anchor = AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)
        if my_verbose:
            print("202309280023")
        return best_anchor, best_model
    ## end function
