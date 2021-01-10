import os
import json
from optparse import OptionParser
from collections import Counter

import pystan
import numpy as np
from scipy.special import expit, logit, softmax

from models.binary_models import basic_binary_model, binary_vigilance_model
from models.categorical_models import basic_categorical_model, categorical_vigilance_model
from models.count_models import basic_poisson_model


def main():
    usage = "%prog labels.jsonlist outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--id-field', type=str, default='id',
                      help='Field with item id: default=%default')
    parser.add_option('--response-field', type=str, default='label',
                      help='Field with labels (responses): default=%default')
    parser.add_option('--annotator-field', type=str, default='annotator',
                      help='Field with annotator name: default=%default')
    parser.add_option('--iter', type=int, default=4000,
                      help='Number of sampling iterations: default=%default')
    parser.add_option('--chains', type=int, default=5,
                      help='Number of sampling chains: default=%default')
    parser.add_option('--no-vigilance', action="store_true", default=False,
                      help='Use worker vigilance term: default=%default')
    parser.add_option('--no-prior', action="store_true", default=False,
                      help='Do not use informative prior on item means: default=%default')
    parser.add_option('--counts', action="store_true", default=False,
                      help='Use a count (Poisson) model instead of categorical: default=%default')

    (options, args) = parser.parse_args()

    infile = args[0]
    outdir = args[1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    id_field = options.id_field
    response_field = options.response_field
    annotator_field = options.annotator_field
    use_vigilance = not options.no_vigilance
    use_prior = not options.no_prior
    use_counts = options.counts

    with open(infile) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]

    item_counter = Counter([line[id_field] for line in lines])
    response_counter = Counter([line[response_field] for line in lines])
    annotator_counter = Counter([line[annotator_field] for line in lines])

    if len(item_counter) > 12:
        print("{:d} items found".format(len(item_counter)))
    else:
        print("Item counts:")
        for k, v in item_counter.most_common():
            print(k, v)

    if len(annotator_counter) > 12:
        print("{:d} annotators found".format(len(annotator_counter)))
    else:
        print("Annotator counts:")
        for k, v in annotator_counter.most_common():
            print(k, v)

    n_items = len(item_counter)
    n_annotators = len(annotator_counter)
    n_response_types = len(response_counter)
    n_total_responses = len(lines)

    # get a sorted list of possibilities
    item_list = sorted(item_counter)
    annotator_list = sorted(annotator_counter)
    response_list = sorted(response_counter)

    if len(response_counter) > 12:
        print("{:d} response types found".format(len(response_counter)))
        if use_counts:
            print("Min/max:", min(response_counter), max(response_counter))
    else:
        print("Responses:")
        for r in response_list:
            print(r, response_counter[r])

    # convert each to a dictionary
    item_dict = dict(zip(item_list, range(len(item_list))))
    annotator_dict = dict(zip(annotator_list, range(len(annotator_list))))
    response_dict = dict(zip(response_list, range(len(response_list))))

    with open(os.path.join(outdir, 'data.json'), 'w') as f:
        json.dump({'item_list': item_list,
                   'annotator_list': annotator_list,
                   'response_list': response_list,
                   'item_dict': item_dict,
                   'annotator_dict': annotator_dict,
                   'response_dict': response_dict},
                  f)

    items = []
    annotators = []
    responses = []

    for line in lines:
        items.append(item_dict[line[id_field]])
        annotators.append(annotator_dict[line[annotator_field]])
        if use_counts:
            responses.append(line[response_field])
        else:
            responses.append(response_dict[line[response_field]])

    if use_counts:
        model = basic_poisson_model

        data = {'n_items': n_items,
                'n_annotators': n_annotators,
                'n_total_responses': n_total_responses,
                'annotator_for_response': [a + 1 for a in annotators],
                'item_for_response': [i + 1 for i in items],
                'responses': responses}

        with open(os.path.join(outdir, 'model_data.json'), 'w') as f:
            json.dump(data, f)

        sm = pystan.StanModel(model_code=model)
        fit = sm.sampling(data=data, iter=options.iter, chains=options.chains)

        item_means = fit.extract('item_means')['item_means']
        n_samples, _ = item_means.shape
        item_std = fit.extract('item_std')['item_std']
        annotator_offsets = fit.extract('annotator_offsets')['annotator_offsets']
        offset_std = fit.extract('offset_std')['offset_std']
        np.savez(os.path.join(outdir, 'samples.npz'),
                 item_means=item_means,
                 item_std=item_std,
                 annotator_offsets=annotator_offsets,
                 offset_std=offset_std)

        # TODO: add vigilance estimates into this
        item_prob_samples = expit(item_means + annotator_offsets.mean(1).reshape((n_samples, 1)))
        est_item_probs = {item: float(np.mean(item_prob_samples[:, i])) for i, item in enumerate(item_list)}

        for i, a in enumerate(annotator_list):
            print(a, np.mean(annotator_offsets[:, i]), np.std(annotator_offsets[:, i]))

        with open(os.path.join(outdir, 'item_probs.json'), 'w') as f:
            json.dump(est_item_probs, f, indent=2)

    elif n_response_types == 2:
        if use_vigilance:
            model = binary_vigilance_model
        else:
            model = basic_binary_model

        data = {'n_items': n_items,
                'n_annotators': n_annotators,
                'n_total_responses': n_total_responses,
                'annotator_for_response': [a + 1 for a in annotators],
                'item_for_response': [i + 1 for i in items],
                'responses': responses}

        with open(os.path.join(outdir, 'model_data.json'), 'w') as f:
            json.dump(data, f)

        sm = pystan.StanModel(model_code=model)
        fit = sm.sampling(data=data, iter=options.iter, chains=options.chains)

        item_means = fit.extract('item_means')['item_means']
        n_samples, _ = item_means.shape
        item_std = fit.extract('item_std')['item_std']
        annotator_offsets = fit.extract('annotator_offsets')['annotator_offsets']
        offset_std = fit.extract('offset_std')['offset_std']
        if use_vigilance:
            vigilance = fit.extract('vigilance')['vigilance']
            np.savez(os.path.join(outdir, 'samples.npz'),
                     item_means=item_means,
                     item_std=item_std,
                     annotator_offsets=annotator_offsets,
                     offset_std=offset_std,
                     vigilance=vigilance)
        else:
            np.savez(os.path.join(outdir, 'samples.npz'),
                     item_means=item_means,
                     item_std=item_std,
                     annotator_offsets=annotator_offsets,
                     offset_std=offset_std)

        # TODO: add vigilance estimates into this
        item_prob_samples = expit(item_means + annotator_offsets.mean(1).reshape((n_samples, 1)))
        est_item_probs = {item: float(np.mean(item_prob_samples[:, i])) for i, item in enumerate(item_list)}

        with open(os.path.join(outdir, 'item_probs.json'), 'w') as f:
            json.dump(est_item_probs, f, indent=2)

    else:
        if use_vigilance:
            model = categorical_vigilance_model
        else:
            model = basic_categorical_model

        if use_prior:
            prior_probs = [response_counter[r] / float(n_total_responses) for r in response_list]
            priors = [float(np.log(p)) for p in prior_probs]
            print("Using priors:")
            for r_i, r in enumerate(response_list):
                print(r,  priors[r_i])
        else:
            priors = [0.] * len(response_counter)

        data = {'n_items': int(n_items),
                'n_annotators': int(n_annotators),
                'n_total_responses': int(n_total_responses),
                'n_levels': int(n_response_types),
                'priors': priors,
                'annotator_for_response': [int(a + 1) for a in annotators],
                'item_for_response': [int(i + 1) for i in items],
                'responses': [int(r + 1) for r in responses]}

        with open(os.path.join(outdir, 'model_data.json'), 'w') as f:
            json.dump(data, f)

        sm = pystan.StanModel(model_code=model)
        fit = sm.sampling(data=data, iter=options.iter, chains=options.chains)

        item_means = fit.extract('item_means')['item_means']
        item_std = fit.extract('item_std')['item_std']
        annotator_offsets = fit.extract('annotator_offsets')['annotator_offsets']
        offset_std = fit.extract('offset_std')['offset_std']
        if use_vigilance:
            vigilance = fit.extract('vigilance')['vigilance']
            np.savez(os.path.join(outdir, 'samples.npz'),
                     item_means=item_means,
                     item_std=item_std,
                     annotator_offsets=annotator_offsets,
                     offset_std=offset_std,
                     vigilance=vigilance)
        else:
            np.savez(os.path.join(outdir, 'samples.npz'),
                     item_means=item_means,
                     item_std=item_std,
                     annotator_offsets=annotator_offsets,
                     offset_std=offset_std)

        mean_annotator_offsets = np.mean(annotator_offsets, 1)
        # TODO: add vigilance estimates into this
        item_prob_samples = softmax(item_means + np.expand_dims(mean_annotator_offsets, 1), axis=2)
        est_item_probs = {item: [float(p) for p in np.mean(item_prob_samples[:, i, :], axis=0)] for i, item in enumerate(item_list)}

        with open(os.path.join(outdir, 'item_probs.json'), 'w') as f:
            json.dump(est_item_probs, f, indent=2)


if __name__ == '__main__':
    main()
