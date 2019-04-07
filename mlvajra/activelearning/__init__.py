from modAL.models import ActiveLearner,Committee,CommitteeRegressor
from modAL.uncertainty import (classifier_entropy,classifier_margin,
                                classifier_uncertainty,uncertainty_sampling,
                                entropy_sampling,entropy)
from modAL.disagreement import (consensus_entropy,consensus_entropy_sampling,
                                KL_max_disagreement,max_disagreement_sampling,
                                max_std_sampling,vote_entropy,vote_entropy_sampling)

from modAL.expected_error import expected_error_reduction
from collections import namedtuple

learners=['ActiveLearner','Committee','CommitteeRegressor']
queryStrategies=['classifier_entropy','classifier_margin', 
                'classifier_uncertainty','uncertainty_sampling',
                'entropy_sampling','entropy','expected_error_reduction']
consensus_list=['consensus_entropy','consensus_entropy_sampling',
                'KL_max_disagreement','max_disagreement_sampling',
                'max_std_sampling','vote_entropy','vote_entropy_sampling']

learner=namedtuple('learner',learners)
queryStrategy=namedtuple('queryStrategy',queryStrategies)
consensus=namedtuple('consensus',consensus_list)
#Learners
learner.ActiveLearner=ActiveLearner
learner.Committee=Committee
learner.CommitteeRegressor=CommitteeRegressor

#qs
queryStrategy.classifier_entropy=classifier_entropy
queryStrategy.classifier_margin=classifier_margin
queryStrategy.classifier_uncertainty=classifier_uncertainty
queryStrategy.entropy=entropy
queryStrategy.entropy_sampling=entropy_sampling
queryStrategy.expected_error_reduction=expected_error_reduction
queryStrategy.uncertainty_sampling=uncertainty_sampling

#consensus
consensus.consensus_entropy=consensus_entropy
consensus.consensus_entropy_sampling=consensus_entropy_sampling
consensus.KL_max_disagreement=KL_max_disagreement
consensus.max_disagreement_sampling=max_disagreement_sampling
consensus.max_std_sampling=max_std_sampling
consensus.vote_entropy=vote_entropy
consensus.vote_entropy_sampling=vote_entropy_sampling

__all__=['learner','queryStrategy','consensus']


