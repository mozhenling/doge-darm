# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import json
import os
import tqdm
from oututils.query import Q

def load_records(path):
    """records are checkpoints info. of steps"""
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),ncols=80, leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records, subalgorithm=None, args_seed_type = 'trial_seed'):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    if subalgorithm is None:
        result = collections.defaultdict(lambda: [])
        for r in records:
            for test_env in r["args"]["test_envs"]:
                # group records by specific keys
                group = (r["args"][args_seed_type],
                         r["args"]["dataset"],
                         r["args"]["algorithm"],
                         test_env)
                result[group].append(r)

        return Q([{args_seed_type: t, "dataset": d, "algorithm": a, "test_env": e,
             "records": Q(r)} for (t,d,a,e),r in result.items()])
    else:
        result = collections.defaultdict(lambda: [])
        for r in records:
            for test_env in r["args"]["test_envs"]:
                # group records by specific keys
                group = (r["args"][args_seed_type],
                         r["args"]["dataset"],
                         r["args"]["algorithm"],
                         r["args"]["sub_algorithm"],
                         test_env)
                result[group].append(r)

        return Q([{args_seed_type: t, "dataset": d, "algorithm": a,'sub_algorithm':s, "test_env": e,
                   "records": Q(r)} for (t, d, a, s, e), r in result.items()])

