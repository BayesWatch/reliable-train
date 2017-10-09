'''Script to cleanup existing checkpoints'''

import os
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from checkpoint import existing_checkpoints
from main import parse, format_model_tag

def main(args):
    # use the same args as main.py to figure out which checkpoints we should be looking at
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1)
    checkpoint_loc = os.path.join(args.scratch, 'checkpoint', model_tag)

    # figure out which checkpoints we don't care about
    existing = existing_checkpoints(checkpoint_loc)
    best_acc, best_loss = 0., 100.
    redundant_checkpoints = []
    for j,(config, info) in enumerate(existing):
        if info['acc'] > best_acc:
            best_acc = info['acc']
            best_by_acc = info['abspath']
        elif info['loss'] < best_loss:
            best_loss = info['loss']
            best_by_loss = info['abspath']
        else:
            # not the best by accuracy or loss so add to list to delete
            redundant_checkpoints.append(info['abspath'])
    # sanity checks
    assert best_by_acc not in redundant_checkpoints
    assert best_by_loss not in redundant_checkpoints

    # list checkpoints that will be deleted
    print("The following checkpoints will be deleted:")
    for checkpoint in redundant_checkpoints:
        print("    %s"%checkpoint)
    print("The following checkpoints will be kept:")
    print("    %s"%best_by_acc)
    print("    %s"%best_by_loss)

    # query if this is OK to continue
    query = input("OK (y/n)?")
    print("Deleting %i files..."%len(redundant_checkpoints))
    if query == 'y':
        for checkpoint in tqdm(redundant_checkpoints):
            os.remove(checkpoint)

if __name__ == '__main__':
    args = parse()
    main(args)

