import functools
import sys

import pandas as pd
from tqdm import tqdm
import wandb
import itertools

from wandb_config import WANDB_PROJECT, WANDB_ENTITY
import logging

logging.basicConfig(level=logging.INFO)

def grid_status(pd_dataframe, map_hp_vals):
    lists = map_hp_vals.values()
    keys = map_hp_vals.keys()
    overview = dict()
    for element in itertools.product(*lists):
        instance = {}
        for i, k in enumerate(keys):
            if k not in pd_dataframe.columns:
                logging.debug(f"{k} not in cols {pd_dataframe.columns}")
                continue
            instance[k] = element[i]
        runs = pd_dataframe.loc[(pd_dataframe[list(instance)] == pd.Series(instance)).all(axis=1)]
        overview[str(instance)] = len(runs)

    return overview


#
# df1 = pd.DataFrame({'lr': [1, 1, 1, 3, 4], 'optim': ["sgd", "sgd", "sgd", "adam", "adam"]})
# cartesian_comp(df1, {"lr": [0, 1, 2, 3], "optim": ["sgd", "adam"]})


def diversity_check(pd_dataframe, hp_name, valeus_to_search):
    val_nums = [pd_dataframe[pd_dataframe[hp_name] == val] for val in valeus_to_search]

    return val_nums


# returns sorted metric values
def best_metric(pd_dataframe, metric_name, top_n=None, sort_invert=True):

    metric_list = []
    for index, row in pd_dataframe.iterrows():
        summ = row['summary']
        if metric_name in summ:
            acc = summ[metric_name]
            metric_list.append(acc)

    if top_n is not None and len(metric_list) < top_n:
        raise AssertionError(f"too little experiments {len(metric_list)} < {top_n} found: {metric_list}")

    if sort_invert:
        ret = sorted(metric_list)[::-1][:top_n]
    else:
        ret = sorted(metric_list)[:top_n]

    return ret[:top_n] if top_n is not None else ret


def filter_data(pf_datafframe, filter):
    return pf_datafframe.loc[(pf_datafframe[list(filter)] == pd.Series(filter)).all(axis=1)]


def load_runs(tags :tuple, mandatory_hp=None, mandatory_m=None, minimum_runtime_s=20, minimum_steps=500, unfold=False):
    
    runs_all = 0

    api = wandb.Api()
    entity, project = WANDB_ENTITY, WANDB_PROJECT
    runs = api.runs(entity + "/" + project)

    columns = ["summary", "config", "name"]
    if mandatory_hp:
        columns += mandatory_hp
    if mandatory_m:
        columns += mandatory_m

    if not unfold:
        df = pd.DataFrame(columns=columns)
    else:
        df = None

    for i_run, run in tqdm(enumerate(runs)):
        logging.debug(f"RUN {run.name}")

        ok = True
        for tag in tags:
            if tag not in run.tags:
                ok = False
                logging.debug(f"--expected tag {tag} not present")
                break
        if not ok:
            continue
        runs_all += 1


        proto = {
            "summary": [run.summary._json_dict],
            "config": [
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')}],
            "name": [run.name]
        }

        js_summary = run.summary._json_dict
        if "_runtime" not in js_summary or "_step" not in js_summary:
            logging.debug("--NOT USING: runtime or steps not present")
            continue
        if js_summary["_runtime"] < minimum_runtime_s:
            logging.debug(f"--NOT USING : {js_summary['_runtime']}s")
            continue
        if js_summary["_step"] < minimum_steps:
            logging.debug(f"--{js_summary['_step']}steps")
            continue

        ## mandatory M
        ok = True
        if mandatory_m:
            for m_m in mandatory_m:
                if m_m not in js_summary:
                    logging.debug(f"--mandatory metric '{m_m}' not found in run ... skipping")
                    ok = False
                    break
                else:
                    proto[m_m] = js_summary[m_m]
        if not ok:
            continue

        ## mandatory HP
        ok = True
        if mandatory_hp:
            js_config = run.config
            for m_hp in mandatory_hp:
                if m_hp not in js_config:
                    logging.debug(f"--mandatory hp '{m_hp}' not found in run ... skipping")
                    ok = False
                    break
                else:
                    proto[m_hp] = run.config[m_hp]
        if not ok:
            continue

        if unfold:
            proto["name"] = proto["name"][0]
            for cfg_name, cfg_value in proto["config"][0].items():
                proto[f"config.{cfg_name}"] = cfg_value
            proto.pop("config")
            for summary_name, summary_value in proto["summary"][0].items():
                proto[f"summary.{summary_name}"] = summary_value
            proto.pop("summary")
            entry = pd.json_normalize(proto)
        else:
            entry = pd.DataFrame.from_dict(proto)

        if df is None:
            df = entry
        else:
            df = pd.concat([df, entry], ignore_index=True)

        logging.debug("--ok")

    print(f"=============================\nUsing {len(df)}/{runs_all} runs\n You can use logging:debug for more info", file=sys.stderr)

    return df


def get_experiments(wandb_data: pd.DataFrame, **config):
    df = wandb_data.copy()
    for name, value in config.items():
        df = df.loc[df[name].isin([value])]
    return df


def has_experiment(wandb_data: pd.DataFrame, **config):
    df = wandb_data.copy()
    for name, value in config.items():
        df = df.loc[df[name].isin([value])]
    return len(df.index) > 0


def has_result_better_than(wandb_data: pd.DataFrame, result_limit: float, result_name: str, **config_filter):
    df = wandb_data.copy()
    for name, value in config_filter.items():
        df = df.loc[df[name].isin([value])]
    df = df.loc[df[result_name] > result_limit]
    return len(df.index) > 0


def has_result_less_than(wandb_data: pd.DataFrame, result_limit: float, result_name: str, **config_filter):
    df = wandb_data.copy()
    for name, value in config_filter.items():
        df = df.loc[df[name].isin([value])]
    df = df.loc[df[result_name] < result_limit]
    return len(df.index) > 0
