import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Union, Dict, Any, Optional, Tuple
from collections import Counter
from sklearn.metrics import classification_report
from itertools import chain, product
from statistics import mean, stdev
from datetime import datetime
from world.world import World


SEP = "__"

AREA_MAP = {
    "NW": "top left",
    "N": "top center",
    "NE": "top right",
    "W": "center left",
    "C": "center",
    "E": "center right",
    "SW": "bottom left",
    "S": "bottom center",
    "SE": "bottom right"
}


def pack_id(elems: List[str]) -> str:
    return SEP.join(elems)

def unpack_id(id: str) -> list:
    return id.split(SEP)

def first_sentence(answ: str) -> str:
    sentences = sent_tokenize(answ.lower()) if answ != "" else []
    return sentences[0] if sentences else ""

def filter_by_ent_size(model_res: Dict[str, Any], worlds: Dict[str, World], ent_size: str) -> Dict[str, Any]:
    filtered = {}
    for id, answ in model_res.items():
        w_id, e_id, *_ = unpack_id(id)
        if worlds[w_id].ents[e_id].size == ent_size:
            filtered[id] = answ
    return filtered

def get_exp_worlds(worlds_per_exp, exp: str, test_set: str, exp_type: str) -> Dict[str, World]:
    if exp not in worlds_per_exp:
        # remove _run_n
        exp = "_".join(exp.split("_")[:-2])
    return worlds_per_exp[exp][test_set][exp_type]


def load_repr(
    exp_name: str,
    folder,
) -> Tuple[Dict[str, Any], Dict[str, World]]:
    with open(os.path.join(folder, exp_name, "repr.json"), "r") as f:
        repr = json.load(f)

    worlds = {}
    for w_id, w_repr in repr.items():
        worlds[w_id] = World.from_repr(w_repr)

    return repr, worlds

def format_results(results: Dict[str, Any], exp_type: str) -> Dict[str, Any]:
    if exp_type != "position_absolute":
        raise Exception(f"exp_type '{exp_type}' not supported.")
    new_res = {}
    for w_id, w in results.items():
        for e_id, e in w.items():
            id = pack_id([w_id, e_id, exp_type])
            new_res[id] = e[exp_type]["pred"]
    return new_res

def load_results(
    exp_name: str,
    exp_type: str,
    models: List[str],
    results_path: str,
) -> Dict[str, Any]:

    results = {}
    for model in os.listdir(results_path):
        if model not in models:
            continue
        file_name = "generation_results.json"
        res_path = os.path.join(results_path, model, exp_name, file_name)
        if os.path.exists(res_path):
            with open(res_path, "r") as f:
                try:
                    results[model] = format_results(json.load(f), exp_type)
                except Exception as e:
                    print(f"Exception e: {e}")
                    print(f"Exception: Results for model '{model}' and '{exp_type}' not found.")
        else:
            print(f"Results for model '{model}' and '{exp_type}' not found. '{res_path}' does not exist")
    return results


def get_lengths(model_res: Dict[str, Any]) -> List[int]:
    return [len(word_tokenize(first_sentence(answ))) for answ in model_res.values()]

def get_lengths_table(
    results,
    experiments: List[str],
    models: List[str],
    test_sets: List[str],
):
    table = []
    exp_type = "position_absolute"
    for exp in experiments: # rows
        for model in models: # row groups
            row = [exp, model]
            for test_set in test_sets:
                if model not in results[exp][test_set][exp_type]:
                    row.append("---")
                    continue
                lengths = get_lengths(results[exp][test_set][exp_type][model])
                row.append(f"{round(mean(lengths))} ± {stdev(lengths):.2f}")
            table.append(row)

    return table


def get_gt(results: Dict[str, Any], worlds: Dict[str, World]) -> Dict[str, Any]:
    gt = {}
    for id in results:
        w_id, e_id, experiment = unpack_id(id)
        gt[id] = AREA_MAP[worlds[w_id].ents[e_id].area]
    return gt

def get_pred(results: Dict[str, Any]) -> Dict[str, Any]:
    values = set(AREA_MAP.values())
    preds = {}
    for id, answ in results.items():
        first_sent = first_sentence(answ)

        found_vals = []
        for val in values:
            if val == "center":
                matched = (
                    "center" in first_sent and
                    all([
                        v not in first_sent
                        for v in ["top center", "bottom center", "center left", "center right"]
                    ])
                )
            else:
                matched = val in first_sent
            if matched:
                found_vals.append(val)
        preds[id] = found_vals
    return preds

def get_classif_report(gt: Dict[str, Any], pred: Dict[str, Any], output_dict=True) -> Union[dict, str]:
    gt_list = []
    pred_list = []

    for id, predictions in pred.items():
        pred_val = "other"
        if len(predictions) == 1:
            pred_val = predictions[0]
        gt_list.append(gt[id])
        pred_list.append(pred_val)

    return classification_report(
        gt_list,
        pred_list,
        output_dict=output_dict,
        zero_division=0.0
    )

def get_other_count(pred: Dict[str, Any]) -> int:
    return len([predictions for predictions in pred.values() if len(predictions) != 1])

def get_results_table(
    results,
    worlds_per_exp,
    experiments: List[str],
    models: List[str],
    test_sets: List[str],
):
    table = []
    exp_type = "position_absolute"

    for exp in experiments:
        for model in models:
            row = [exp, model]
            for test_set in test_sets:
                if model not in results[exp][test_set][exp_type]:
                    row.append("---")
                    continue
                model_res = results[exp][test_set][exp_type][model]

                gt = get_gt(model_res, get_exp_worlds(worlds_per_exp, exp, test_set, exp_type))
                pred = get_pred(model_res)

                report = get_classif_report(gt, pred)
                row.append(report["accuracy"])

            table.append(row)
    return table

def get_other_table(
    results,
    worlds_per_exp,
    experiments: List[str],
    models: List[str],
    test_sets: List[str],
):
    table = []
    exp_type = "position_absolute"

    for exp in experiments:
        for model in models:
            row = [exp, model]

            for test_set in test_sets:
                if model not in results[exp][test_set][exp_type]:
                    row.append("---")
                    continue
                model_res = results[exp][test_set][exp_type][model]

                gt = get_gt(model_res, get_exp_worlds(worlds_per_exp, exp, test_set, exp_type))
                pred = get_pred(model_res)

                row.append(100*get_other_count(pred)/len(gt))

            table.append(row)
    return table


def get_report_at_pos(
    results: Dict[str, Any],
    worlds: Dict[str, World],
    pos: Tuple[int, int],
    output_dict: bool = True,
) -> Tuple[Union[dict, str], Dict[str, Any], Dict[str, Any]]:

    # keep only the results of the entities at the given position
    filtered_res = {}
    for id, answ in results.items():
        w_id, e_id, experiment = unpack_id(id)
        if worlds[w_id].ents[e_id].position == pos:
            filtered_res[id] = answ

    gt = get_gt(filtered_res, worlds)
    pred = get_pred(filtered_res)
    gt = filter_by_ent_size(gt, worlds, "large")
    pred = filter_by_ent_size(pred, worlds, "large")

    report = get_classif_report(gt, pred, output_dict=output_dict)

    return report, gt, pred


def save_plot(fig, save_path: str, test_set: str, exp_type: str, model: str):
    os.makedirs(save_path, exist_ok=True)
    for filetype in [".png", ".pdf"]:
        filename = SEP.join([test_set, exp_type, model, filetype])
        fig.savefig(os.path.join(save_path, filename), bbox_inches="tight")

def draw_grid(
    ax,
    grid: Optional[str] = None,
    dist: float = 0.0375,
    linewidth: float = 1.5,
    offset: float = 0.0,
    single_grid: bool = True
):
    show_x_grid = grid in ["all", "x"]
    show_y_grid = grid in ["all", "y"]

    n_rows = 9

    if show_x_grid:
        if single_grid:
            for pos in np.arange(3+offset, n_rows-1, 3):
                ax.axvline(pos, linestyle="--", linewidth=linewidth, color="black")
        else:
            for pos in np.arange(3+offset, n_rows-1, 3):
                ax.axvline(pos-dist, linestyle="--", linewidth=linewidth, color="black")
                ax.axvline(pos, linestyle="--", linewidth=linewidth, color="white")
                ax.axvline(pos+dist, linestyle="--", linewidth=linewidth, color="black")
    if show_y_grid:
        if single_grid:
            for pos in np.arange(3+offset, n_rows-1, 3):
                ax.axhline(pos, linestyle="--", linewidth=linewidth, color="black")
        else:
            for pos in np.arange(3+offset, n_rows-1, 3):
                ax.axhline(pos-dist, linestyle="--", linewidth=linewidth, color="black")
                ax.axhline(pos, linestyle="--", linewidth=linewidth, color="white")
                ax.axhline(pos+dist, linestyle="--", linewidth=linewidth, color="black")
    if grid is not None and single_grid:
        ax.grid(which="minor", linestyle="--", linewidth=linewidth, color="black")

def get_acc_by_pos(
    model_res: Dict[str, Any],
    worlds: Dict[str, World],
) -> List[List[float]]:
    prop_by_pos = [[0 for j in range (9)] for i in range(9)]

    for pos in product(range(9), repeat=2):
        r, c = pos
        report, *_ = get_report_at_pos(model_res, worlds, pos)
        prop_by_pos[r][c] = report["accuracy"]
    return prop_by_pos

def plot_by_pos(
    perf_by_pos: List[List[float]],
    exp_type: str,
    model: str,
    test_set: str,
    save_path: str,
    grid: Optional[str] = None,
    single_grid: bool = True,
    plot_only: bool = True,
):
    plt.clf()
    plt.close()
    fig, ax = plt.subplots(figsize=(5.5,5.5), frameon=(not plot_only))

    cax = ax.imshow(perf_by_pos, cmap="viridis", aspect="auto")

    ax.set_aspect("equal")

    if not plot_only:
        fig.colorbar(cax)

    n_rows, n_cols = len(perf_by_pos), len(perf_by_pos[0])

    max_val = max([max(r) for r in perf_by_pos])
    min_val = min(min(r) for r in perf_by_pos)
    mid = max_val - ((max_val - min_val)/2)
    if max_val == min_val:
        cax.set_clim(0, max_val)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{int(perf_by_pos[i][j]*100)}", ha='center', va='center',
                    color="white" if perf_by_pos[i][j] < mid else "black")

    if plot_only:
        ax.tick_params(bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    else:
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(9))
        ax.set_yticks(np.arange(9))

    draw_grid(ax, grid=grid, offset=-0.5, single_grid=single_grid)

    save_plot(fig, save_path, test_set, exp_type, model)

def gen_results_plots(
    results,
    worlds_per_exp,
    experiments: List[str],
    models: List[str],
    test_sets: List[str],
    save_path = "./analysis/by_pos_percentage",
):
    exp_type = "position_absolute"
    for exp in experiments:
        for model in models:
            for test_set in test_sets:
                if model not in results[exp][test_set][exp_type]:
                    continue

                print(f"----- exp: {exp}; test_set: {test_set}; exp_type: {exp_type}; model: {model}")

                perf_by_pos = get_acc_by_pos(
                    results[exp][test_set][exp_type][model],
                    get_exp_worlds(worlds_per_exp, exp, test_set, exp_type),
                )
                plot_by_pos(
                    perf_by_pos,
                    exp_type,
                    model,
                    test_set,
                    os.path.join(save_path, exp),
                    grid="all",
                    single_grid=False,
                )

def merge_res_by_run(
    results,
    experiments: List[str],
    experiments_with_runs: List[str],
    models: List[str],
    test_sets: List[str],
    filtered_runs_ids: dict,
):
    exp_type = "position_absolute"
    results_merged = {}
    for exp_no_run in experiments:
        results_merged[exp_no_run] = {}
        for exp in experiments_with_runs:
            if exp_no_run not in exp:
                continue

            if exp_no_run == exp:
                run_str = ""
            else:
                run_str = exp.split(f"{exp_no_run}")[1]

            for test_set in test_sets:
                if test_set not in results_merged[exp_no_run]:
                    results_merged[exp_no_run][test_set] = {exp_type: {}}
                for model in models:
                    if len(filtered_runs_ids[exp_no_run][model]) > 1 and int(exp[-1]) not in filtered_runs_ids[exp_no_run][model]:
                        # if run discarded do not add results
                        continue
                    if model not in results[exp][test_set][exp_type]:
                        continue
                    model_res = {f"{id}{run_str}": answ for id, answ in results[exp][test_set][exp_type][model].items()}
                    if model not in results_merged[exp_no_run][test_set][exp_type]:
                        results_merged[exp_no_run][test_set][exp_type][model] = model_res
                    else:
                        results_merged[exp_no_run][test_set][exp_type][model].update(model_res)
    return results_merged


def get_most_frequent(
    model_res: Dict[str, Any],
    worlds: Dict[str, World],
):
    max_pred_by_pos = [[0 for j in range (9)] for i in range(9)]
    max_pred_ratio_by_pos = [[0 for j in range (9)] for i in range(9)]

    for pos in product(range(9), repeat=2):
        r, c = pos
        report, gt, pred = get_report_at_pos(model_res, worlds, pos)
        pred_list = []
        for predictions in pred.values():
            pred_val = "other"
            if len(predictions) == 1:
                pred_val = predictions[0]
            pred_list.append(pred_val)

        pred_counts = Counter(pred_list)
        try:
            max_pred = max(pred_counts, key=pred_counts.get)
            max_pred_ratio = pred_counts[max_pred]/sum(pred_counts.values())
            max_pred_ratio_by_pos[r][c] = max_pred_ratio
        except:
            max_pred = "other"
            max_pred_ratio_by_pos[r][c] = 0
        max_pred_by_pos[r][c] = max_pred

    return max_pred_by_pos, max_pred_ratio_by_pos

def plot_maj_voting(
    maj_classes: List[List[str]],
    confidence: List[List[int]],
    model: str,
    exp_type: str,
    test_set: str,
    save_path: str,
    grid: Optional[str],
    single_grid: bool = True,
    plot_only: bool = False,
    show_confidence: bool = True,
    use_alpha: bool = True,
    use_tab10: bool = False,
    show_patterns: bool = False,
    cell_border_width: int = 0,
):
    plt.clf()
    plt.close()
    if plot_only:
        fig, ax = plt.subplots(frameon=False)
    else:
        fig, ax = plt.subplots()

    patterns = ["/", "x", "\\" ,"-" , "+" , "|" , "o", "O", ".", "*"]

    classes = AREA_MAP.values()

    if use_alpha:
        if use_tab10:
            colors = matplotlib.colormaps["tab10"].colors
        else:
            colors = [
                "#D62728",
                "#3C88D0",
                "#7F7F7F",
                "#3CB44B",
                "#911EB4",
                "#F58231",
                "#B39C00",
                "#F032E6",
                "#9A6324",
            ]

        val_to_color = {
            val: (color, patt)
            for val, color, patt in zip(classes, colors, patterns)
        }
        val_to_color["other"] = ("black", "")
    else:
        tab20 = plt.get_cmap("tab20")
        color_pairs = [
            (tab20(i+1), tab20(i))
            for i in range(0,20,2)
        ]
        val_to_color = {
            val: (matplotlib.colors.LinearSegmentedColormap.from_list(val, c_pair), patt)
            for val, c_pair, patt in zip(classes, color_pairs, patterns)
        }
        val_to_color["other"] = matplotlib.colors.LinearSegmentedColormap.from_list("other", ["white", "black"])

    n_rows, n_cols = len(maj_classes), len(maj_classes[0])
    for i in range(n_rows):
        for j in range(n_cols):
            val = maj_classes[i][j]
            color, patt = val_to_color[val]
            if show_confidence:
                ratio = confidence[i][j]
            else:
                ratio = 1.0

            if use_alpha:
                alpha = ratio
                edgecolor = color
            else:
                alpha = 1.0
                edgecolor = color(1.0)
                color = color(ratio)

            if not show_patterns:
                patt = ""
            else:
                edgecolor = "black"

            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1, facecolor=color, edgecolor=edgecolor,
                    linewidth=cell_border_width, alpha=alpha, hatch=patt
                )
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    if plot_only:
        ax.tick_params(bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    else:
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(0.5, n_cols, 1))
        ax.set_yticks(np.arange(0.5, n_rows, 1))
        ax.set_xticklabels(range(n_cols))
        ax.set_yticklabels(range(n_rows))

    draw_grid(ax, grid=grid, single_grid=single_grid)

    ax.invert_yaxis()
    ax.set_aspect("equal")

    if not plot_only:
        if use_alpha:
            legend_elements = [
                plt.Line2D([0], [0], marker="s", color="w", label=f"{val}",
                           markerfacecolor=color, markersize=10)
                for val, (color, patt) in val_to_color.items()
            ]
        else:
            legend_elements = [
                plt.Line2D([0], [0], marker="s", color="w", label=f"{val}",
                           markerfacecolor=color(1.0), markersize=10)
                for val, (color, patt) in val_to_color.items()
            ]

        ax.legend(handles=legend_elements, loc="upper left", title="Predictions", bbox_to_anchor=(1.02,1), borderaxespad=0)

        plt.title(f"Position according to '{model}'")

    save_plot(fig, save_path, test_set, exp_type, model)

def gen_pos_plots(
    results,
    worlds_per_exp,
    experiments: List[str],
    models: List[str],
    test_sets: List[str],
    save_path = "./analysis/areas_for_models",
):
    exp_type = "position_absolute"
    for exp in experiments:
        for test_set in test_sets:
            for model in models:
                if model not in results[exp][test_set][exp_type]:
                    continue
                print(f"----- exp: {exp}; test_set: {test_set}; exp_type: {exp_type}; model: {model}")

                maj_classes, confidence = get_most_frequent(
                    results[exp][test_set][exp_type][model],
                    get_exp_worlds(worlds_per_exp, exp, test_set, exp_type),
                )
                plot_maj_voting(
                    maj_classes,
                    confidence,
                    model,
                    exp_type,
                    test_set,
                    os.path.join(save_path, exp),
                    plot_only=True,
                    show_confidence=False,
                    cell_border_width=0,
                    grid="all",
                )


def save_results_csv(pd_table: pd.DataFrame, save_folder: str, file_name: str):
    csv_results = os.path.join(save_folder, file_name)
    os.makedirs(save_folder, exist_ok=True)
    if os.path.isfile(csv_results):
        old_res = pd.read_csv(csv_results, index_col=False)
        n_rows, n_cols = old_res.shape
        new_keys = set(pd_table.keys())
        old_keys = set(old_res.keys())
        if not (len(new_keys) == len(old_keys) and len(new_keys.difference(old_keys)) == 0):
            old_res = old_res.assign(n_runs=pd.Series([1 for _ in range(n_rows)]).values)
            old_res = old_res.assign(civet_test_std=pd.Series([None for _ in range(n_rows)]).values)
            old_res = old_res.assign(coco_test_std=pd.Series([None for _ in range(n_rows)]).values)
            old_res = old_res.reindex(columns=["experiment", "model", "civet_test",  "civet_test_std","coco_test", "coco_test_std", "n_runs"])
            old_res.to_csv(csv_results, mode="w", header=True)
    pd_table.to_csv(csv_results, mode="a", header=not os.path.exists(csv_results))


def update_table_runs(experiments: List[str], exp_with_runs: List[str], table: List[list], columns: List[str], test_sets: List[str], filter: bool = False) -> Tuple[List[list], List[str], Dict[str, Any]]:
    runs_avg_table = []

    map = {}
    for exp_r in exp_with_runs:
        if exp_r in experiments:
            map[exp_r] = exp_r
        else:
            for exp in experiments:
                if exp == "_".join(exp_r.split("_")[:-2]):
                    map[exp_r] = exp

    results = {}
    for row in table:
        exp_name, model, *test_sets_res = row
        exp = map[exp_name]
        if exp not in results:
            results[exp] = {}
        if model not in results[exp]:
            results[exp][model] = {}

        for i, test_set in enumerate(test_sets):
            if test_set not in columns:
                continue
            if test_set not in results[exp][model]:
                results[exp][model][test_set] = []
            results[exp][model][test_set].append(test_sets_res[i])


    runs_avg_columns = ["experiment", "model"] + list(chain(*[[test_set, f"{test_set}_std"] for test_set in test_sets if test_set in columns])) + ["n_runs"]
    filtered_keep = {}
    for exp, models in results.items():
        if exp not in filtered_keep:
            filtered_keep[exp] = {}
        for model, test_sets_res in models.items():
            if model not in filtered_keep[exp]:
                filtered_keep[exp][model] = {}

            row = [exp, model]

            test_sets_res_filtered = test_sets_res
            if filter and "civet_test" in test_sets and "civet_test" in test_sets_res:
                test_sets_res_filtered = {}
                civet_res = test_sets_res["civet_test"]
                keep_run = [not isinstance(r, str) and r >= 0.12 for r in civet_res]

                keep_run_ids = [id for id, r in enumerate(civet_res) if not isinstance(r, str) and r >= 0.12]

                filtered_keep[exp][model] = keep_run_ids

                for test_set, res in test_sets_res.items():
                    test_sets_res_filtered[test_set] = [r for r, keep in zip(res, keep_run) if keep]
                assert all([r >= 0.12 for r in test_sets_res_filtered["civet_test"]])

            for test_set in test_sets:
                if test_set in test_sets_res:
                    res = test_sets_res_filtered[test_set]
                    if len(res) > 0 and not isinstance(res[0], str):
                        if len(res) > 1:
                            row.extend([mean(res), stdev(res)])
                        else:
                            row.extend([res[0], None])
                    else:
                        row.extend(["---", "---"])
                else:
                    row.extend(["---", "---"])

            row.append(len(list(test_sets_res_filtered.values())[0]))
            runs_avg_table.append(row)

    return runs_avg_table, runs_avg_columns, filtered_keep

def print_formatted_results(df_res, test_sets: List[str]):
    for test_set in test_sets:
        test_set_std = f"{test_set}_std"

        # remove "---"
        df_res[test_set] = pd.to_numeric(df_res[test_set], errors="coerce")
        df_res[test_set_std] = pd.to_numeric(df_res[test_set_std], errors="coerce")

        df_res[test_set] = df_res[test_set].apply(lambda x: f"{x:.2f}")
        df_res[test_set_std] = df_res[test_set_std].apply(lambda x: f"{x:.2f}")
        df_res[test_set] = df_res[test_set] + " ± " + df_res[test_set_std]

        df_res.drop([test_set_std], axis=1, inplace=True)

    df_res["index"] = df_res.index
    df_res = df_res.sort_values(by=["model", "index"])
    df_res = df_res.drop("index", axis=1)
    print(df_res)


def analyse_results(
    models: List[str] = ["llava-next-7b"],
    test_sets: List[str] = ["civet_test", "coco_test"],
    exp_types: List[str] = ["position_absolute"],
    experiments: List[str] = ["baseline", "streamline_test"],
    data_folder: str = "./data",
    results_folder: str = "./output",
    save_results: bool = True,
    compute_average: bool = True,
    exclude_list: List[str] = [],
    filter: bool = False,
    name: Optional[str] = None,
):
    print("Loading representations...")
    print("experiments: ", experiments)
    worlds_per_exp: Dict[str, Dict[str, Dict[str, Dict[str, World]]]] = {
        exp: {
            test_set: {
                exp_type: load_repr(test_set, data_folder)[1]
                for exp_type in exp_types
            }
            for test_set in test_sets
        }
        for exp in experiments
    }

    print("Loading results...")
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    exp_with_runs = []
    for exp in experiments:
        exp_root = os.path.join(results_folder, models[0], exp)
        exp_dirs = []
        for exp_file in os.listdir(exp_root):
            if os.path.isdir(os.path.join(exp_root, exp_file)):
                exp_dirs.append(exp_file)
        n_runs = 1
        if "run_0" in exp_dirs:
            n_runs = len(exp_dirs)

        for i in range(n_runs):
            run = f"run_{i}"
            exp_name = exp
            if n_runs > 1:
                exp_name = f"{exp}_{run}"
            if exp_name in exclude_list:
                continue
            exp_with_runs.append(exp_name)
            print(f"Loading {exp_name}")

            results[exp_name] = {}
            for test_set in test_sets:
                results[exp_name][test_set] = {}
                for exp_type in exp_types:
                    try:
                        if n_runs > 1:
                            results[exp_name][test_set][exp_type] = load_results(os.path.join(exp, run, test_set), exp_type, models, results_folder)
                        else:
                            results[exp_name][test_set][exp_type] = load_results(os.path.join(exp,test_set), exp_type, models, results_folder)
                    except Exception as e:
                        print(f"Missing {exp_type} {exp} {test_set}")
    print(experiments)
    print(exp_with_runs)

    # tables
    tmp_folder = os.path.join(results_folder, "single_outputs")
    analysis_name = name
    if name is None:
        analysis_name = '_'.join([str(datetime.now()), *experiments, *test_sets, *models])

    # answer lengths
    lengths_table = get_lengths_table(results, experiments=exp_with_runs, models=models, test_sets=test_sets)
    pd_lengths_table = pd.DataFrame(lengths_table, columns=["experiment", "model"] + test_sets)
    if save_results:
        try:
            save_results_csv(pd_lengths_table, results_folder, "lengths.csv")
        except Exception as e:
            print(f"Could not save to csv: {e}")
        save_results_csv(pd_lengths_table, tmp_folder, f"{analysis_name}_lengths.csv")
    print("-"*100)
    print("Answer Lengths")
    print(pd_lengths_table.round(2))

    # all results
    results_table = get_results_table(results, worlds_per_exp, experiments=exp_with_runs, models=models, test_sets=test_sets)
    columns = ["experiment", "model"] + test_sets
    print("-"*100)
    has_runs = compute_average and len(exp_with_runs) > len(experiments)
    if has_runs:
        results_table, columns, filtered_runs_ids = update_table_runs(experiments, exp_with_runs, results_table, columns, test_sets, filter)
    pd_results_table = pd.DataFrame(results_table, columns=columns)
    if save_results:
        try:
            save_results_csv(pd_results_table, results_folder, "results.csv")
            save_results_csv(pd_results_table, tmp_folder, f"{analysis_name}_results.csv")
        except Exception as e:
            print(f"Could not save to csv: {e}")
    print("-"*100)
    print("Results")
    columns = set(pd_results_table.keys()).difference(set(["experiment", "model", "n_runs"]))
    for col in columns:
        pd_results_table[col] = pd.to_numeric(pd_results_table[col], errors="coerce")
        pd_results_table[col] = pd_results_table[col].round(2)
        pd_results_table[col] = pd_results_table[col].fillna("---")
    print(pd_results_table)
    if compute_average:
        print_formatted_results(pd_results_table, test_sets)

    # 'other' %
    other_table = get_other_table(results, worlds_per_exp, experiments=exp_with_runs, models=models, test_sets=test_sets)
    pd_other_table = pd.DataFrame(other_table, columns=["experiment", "model"] + test_sets)
    if save_results:
        save_results_csv(pd_other_table, results_folder, "other_perc.csv")
        save_results_csv(pd_other_table, tmp_folder, f"{analysis_name}_other_perc.csv")
    print("-"*100)
    print("Other %")
    print(pd_other_table.round(2))

    # plots
    if save_results:
        exp_list = exp_with_runs
        if has_runs:
            results = merge_res_by_run(results, experiments=experiments, experiments_with_runs=exp_with_runs, models=models, test_sets=test_sets, filtered_runs_ids=filtered_runs_ids)
            exp_list = experiments

        # accuracy by position
        gen_results_plots(results, worlds_per_exp, experiments=exp_list, models=models, test_sets=test_sets)

        # position according to models
        gen_pos_plots(results, worlds_per_exp, experiments=exp_list, models=models, test_sets=test_sets)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", type=str, nargs="*", default=["llava-next-7b", "molmo-o-7b", "qwen2-vl-7b"], help="Names of the models.")
    parser.add_argument("--test_sets", type=str, nargs="*", default=["civet_test", "coco_test"], help="Test sets to use for evaluation.")
    parser.add_argument("--experiments", type=str, nargs="*", default=["baseline"], help="Names of the experiments.")
    parser.add_argument("--data_folder", type=str, default="./data", help="Folder containing the test sets (with their 'repr.json').")
    parser.add_argument("--results_folder", type=str, default="./output", help="Folder containing the results (<model>/<experiment>/<test_set>/generation_results.json).")
    parser.add_argument("--save_results", action="store_true", help="Save results.")
    parser.add_argument("--no_average", action="store_true", help="Do not average results over multiple runs.")
    parser.add_argument("--exclude", nargs="*", default=[], help="List of outlier runs to exclude.")
    parser.add_argument("--filter", action="store_true", help="Remove outliers, i.e., runs with < 12%% accuracy on 'civet_test'.")
    parser.add_argument("--name", type=str, default=None, help="Name of analysis.")

    args = parser.parse_args()

    analyse_results(
        models=args.models,
        test_sets=args.test_sets,
        experiments=args.experiments,
        data_folder=args.data_folder,
        results_folder=args.results_folder,
        save_results=args.save_results,
        compute_average=not args.no_average,
        exclude_list=args.exclude,
        filter=args.filter,
        name=args.name
    )

if __name__ == "__main__":
    main()
