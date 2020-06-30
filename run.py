import os
import json
from optparse import OptionParser
from collections import Counter

import pystan
import numpy as np
import matplotlib.pyplot as plt

from models.binary_model import model as binary_model


def main():
    usage = "%prog labels.jsonlist outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--id-field', type=str, default='id',
                      help='Field with item id: default=%default')
    parser.add_option('--response-field', type=str, default='label',
                      help='Field with labels (responses): default=%default')
    parser.add_option('--annotator-field', type=str, default='annotator',
                      help='Field with annotator name: default=%default')
    parser.add_option('--iter', type=int, default=2000,
                      help='Number of sampling iterations: default=%default')
    parser.add_option('--chains', type=int, default=3,
                      help='Number of sampling chains: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    infile = args[0]
    outdir = args[1]

    id_field = options.id_field
    response_field = options.response_field
    annotator_field = options.annotator_field

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

    if len(response_counter) > 12:
        print("{:d} response types found".format(len(response_counter)))
    else:
        print("Responses:")
        for k, v in response_counter.most_common():
            print(k, v)

    n_items = len(item_counter)
    n_annotators = len(annotator_counter)
    n_response_types = len(response_counter)
    n_total_responses = len(lines)

    item_dict = dict(zip(sorted(item_counter), range(len(item_counter))))
    annotator_dict = dict(zip(sorted(annotator_counter), range(len(annotator_counter))))
    response_dict = dict(zip(sorted(response_counter), range(len(response_counter))))

    items = []
    annotators = []
    responses = []

    for line in lines:
        items.append(item_dict[line[id_field]])
        annotators.append(annotator_dict[line[annotator_field]])
        responses.append(response_dict[line[response_field]])

    if n_response_types == 2:
        model = binary_model
    else:
        raise NotImplementedError("Only the binary model is currently implemented")

    data = {'n_items': n_items,
            'n_annotators': n_annotators,
            'n_total_responses': n_total_responses,
            'annotator_for_response': [a + 1 for a in annotators],
            'item_for_response': [i + 1 for i in items],
            'responses': responses}

    with open(os.path.join(outdir, 'data.json'), 'w') as f:
        json.dump(data, f)

    sm = pystan.StanModel(model_code=model)
    fit = sm.sampling(data=data, iter=options.iter, chains=options.chains)

    item_means = fit.extract('item_means')['item_means']
    item_std = fit.extract('item_std')['item_std']
    annotator_offsets = fit.extract('annotator_offsets')['annotator_offsets']
    offset_std = fit.extract('offset_std')['offset_std']
    vigilance = fit.extract('vigilance')['vigilance']

    np.savez(os.path.join(outdir, 'samples.npz'),
             item_means=item_means,
             item_std=item_std,
             annotator_offsets=annotator_offsets,
             offset_std=offset_std,
             vigilance=vigilance)

    fig, ax = plt.subplots()
    for a in range(n_annotators):
        ax.plot(vigilance[:, a])
    plt.savefig(os.path.join(outdir, 'vigilance.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
