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
from typing import Type
from typing import cast 
from typing import Any 

Predicate = tuple[int, str, int]
Rule = list[Predicate]
Classifier = Callable[[np.ndarray], np.ndarray]
Surrogate = Type[compose.Pipeline]
SampleFn = Callable[
        [Rule, int, bool, Surrogate | None, bool], 
        tuple[np.ndarray, np.ndarray, np.ndarray]]

class AnchorBaseBeam(anchor_base.AnchorBaseBeam):
    @staticmethod
    def get_sample_fns(
            sample_fn       : SampleFn,
            tuples          : Rule,
            state,
            surrogate_models: list[Surrogate],
            update_model    : bool):
        
        # each sample fn returns number of positives
        sample_fns = []

        def complete_sample_fn(
                t       : Predicate, 
                n       : int, 
                model   : Surrogate):
            
            # *****************************************************************
            # Take a sample satisfying the tuple t.
            raw_data, data, labels = sample_fn(
                    list(t),
                    n, 
                    True,
                    model,
                    update_model)
            # *****************************************************************

            current_idx = state['current_idx']
            # idxs = range(state['data'].shape[0], state['data'].shape[0] + n)
            idxs = range(current_idx, current_idx + n)

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
            return labels.sum()
        ## end function

        for t, m in zip(tuples, surrogate_models):
            sample_fns.append(
                    lambda n, t=t, model=m: complete_sample_fn(t, n, model))
        return sample_fns
    ## end function

    @staticmethod
    def anchor_beam(sample_fn, delta=0.05, epsilon=0.1, batch_size=10,
                    min_shared_samples=0, desired_confidence=1, beam_size=1,
                    verbose=False, epsilon_stop=0.05, min_samples_start=0,
                    max_anchor_size=None, verbose_every=1,
                    stop_on_first=False, coverage_samples=10000):
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        _, coverage_data, _ = sample_fn([], coverage_samples, compute_labels=False)

        # 最小個数だけサンプリング
        raw_data, data, labels = sample_fn([], max(1, min_samples_start))

        # anchorの精度の期待値
        if len(labels) > 0:
            mean = np.mean(labels)
        else:
            mean = 0

        beta = np.log(1. / delta)

        # anchorの精度の信頼下限
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])

        while mean > desired_confidence and lb < desired_confidence - epsilon:
            # 新しくサンプリング
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            # 新しいサンプルを結合
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            # 精度の平均と信頼下限を更新
            mean = labels.mean()
            lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        ##

        # 信頼下限が制約を満たすならば終了
        if lb > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor

        # メモリを確保してゼロ埋め
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]),
                                         data.dtype)))
        raw_data = np.vstack(
            (raw_data, np.zeros((prealloc_size, raw_data.shape[1]),
                                raw_data.dtype)))
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))

        n_features = data.shape[1]
        state = {'t_idx': collections.defaultdict(lambda: set()),
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
        best_of_size = {0: []}
        best_coverage = -1
        best_tuple = ()
        best_model = None
        t = 1
        if max_anchor_size is None:
            max_anchor_size = n_features

        def init_surrogate_models(num_models):
            models = []
            for i in range(num_models):
                preprocessor = preprocessing.StandardScaler()
                model = linear_model.LogisticRegression()
                pipeline = compose.Pipeline(preprocessor, model)
                models.append(pipeline)
            return models
        ## end function

        while current_size <= max_anchor_size:

            # Call 'GenerateCands' and get INDEXes to the candidates.
            tuples = AnchorBaseBeam.make_tuples(
                best_of_size[current_size - 1], state)
            tuples = [x for x in tuples
                      if state['t_coverage'][x] > best_coverage]
            if len(tuples) == 0:
                break

            surrogate_models = init_surrogate_models(len(tuples)) 
            sample_fns = AnchorBaseBeam.get_sample_fns(
                    sample_fn, tuples, state, surrogate_models, True)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            # print tuples, beam_size

            # Call 'B-BestCands' and get INDEXes to the best candidates.
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                top_n=min(beam_size, len(tuples)),
                verbose=verbose, verbose_every=verbose_every)

            # Get candidates from their indexes
            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            if verbose:
                print('Best of size ', current_size, ':')
            # print state['data'].shape[0]

            stop_this = False

            # t --- a tuple in best candidates
            # i --- an INDEX to the tuple t
            for i, t in zip(chosen_tuples, best_of_size[current_size]):
                # I can choose at most (beam_size - 1) tuples at each step,
                # and there are at most n_feature steps
                beta = np.log(1. /
                              (delta / (1 + (beam_size - 1) * n_features)))
                # beta = np.log(1. / delta)
                # if state['t_nsamples'][t] == 0:
                #     mean = 1
                # else:

                # Update confidence interval and coverage of the tuple t.
                mean = state['t_positives'][t] / state['t_nsamples'][t]
                lb = AnchorBaseBeam.dlow_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                ub = AnchorBaseBeam.dup_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]
                if verbose:
                    print(i, mean, lb, ub)


                # Judge whether the tuple t is an anchor or not.
                while ((mean >= desired_confidence and
                       lb < desired_confidence - epsilon_stop) or
                       (mean < desired_confidence and
                        ub >= desired_confidence + epsilon_stop)):
                    # print mean, lb, state['t_nsamples'][t]
                    sample_fns[i](batch_size)
                    mean = state['t_positives'][t] / state['t_nsamples'][t]
                    lb = AnchorBaseBeam.dlow_bernoulli(
                        mean, beta / state['t_nsamples'][t])
                    ub = AnchorBaseBeam.dup_bernoulli(
                        mean, beta / state['t_nsamples'][t])
                ## end while
                if verbose:
                    print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' % (t, mean, lb, ub, coverage, state['t_nsamples'][t]))
                # If the tuple t is the anchor with the provisionally best
                # coverage, update 'best_tuple' and 'best_model'.
                if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                    if verbose:
                        print('Found eligible anchor ', t, 'Coverage:',
                              coverage, 'Is best?', coverage > best_coverage)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tuple = t
                        best_model = copy.deepcopy(surrogate_models[i])
                        if best_coverage == 1 or stop_on_first:
                            stop_this = True
                ## end if
            ## end for

            if stop_this:
                break
            current_size += 1
        ## end while

        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            if verbose:
                print('Could not find an anchor, now doing best of each size')
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            # tuples = best_of_size[current_size - 1]

            surrogate_models = init_surrogate_models(len(tuples)) 
            sample_fns = AnchorBaseBeam.get_sample_fns(
                    sample_fn, tuples, state, surrogate_models, True)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            # print tuples, beam_size
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                1, verbose=verbose)
            best_tuple = tuples[chosen_tuples[0]]
            best_model = surrogate_models[chosen_tuples[0]]
        ## end if

        # return best_tuple, state

        best_anchor = AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)
        print("202309261115")
        return best_anchor, best_model
    ## end function
