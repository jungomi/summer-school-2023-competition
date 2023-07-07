import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lavd
import torch
import torchmetrics.functional.text as metrics
from halo import Halo


def summarise_results(
    results: Dict,
    invalid: Dict[str, List[str]],
    gt_sources: List[str],
    pred_sources: List[str],
    exclude_missing: bool = False,
) -> Dict:
    aggregated_metrics: Dict = {
        p_s: {g_s: dict(cer=[], wer=[], missing=[], correct=[]) for g_s in gt_sources}
        for p_s in pred_sources
    }
    for result_name, result in results.items():
        gt_name = result["source"]
        for pred_name in pred_sources:
            pred = result["preds"].get(pred_name)
            if pred is None:
                aggregated_metrics[pred_name][gt_name]["missing"].append(result_name)
                if not exclude_missing:
                    # When it's missing, the error rates are set to 1.0 because that is
                    # error rate compared to an empty string (all insertions).
                    # This will be ignored when exclude_missing=True
                    aggregated_metrics[pred_name][gt_name]["cer"].append(1.0)
                    aggregated_metrics[pred_name][gt_name]["wer"].append(1.0)
                    aggregated_metrics[pred_name][gt_name]["correct"].append(0.0)
            else:
                aggregated_metrics[pred_name][gt_name]["cer"].append(pred["cer"])
                aggregated_metrics[pred_name][gt_name]["wer"].append(pred["wer"])
                aggregated_metrics[pred_name][gt_name]["correct"].append(
                    pred["cer"] == 0.0
                )
    stats = {
        pred_name: {
            gt_name: dict(
                cer=torch.mean(torch.tensor(metrics["cer"], dtype=torch.float)).item(),
                wer=torch.mean(torch.tensor(metrics["wer"], dtype=torch.float)).item(),
                missing=len(metrics["missing"]),
                correct=torch.mean(
                    torch.tensor(metrics["correct"], dtype=torch.float)
                ).item(),
            )
            for gt_name, metrics in preds.items()
        }
        for pred_name, preds in aggregated_metrics.items()
    }
    return stats


def split_named_arg(arg: str) -> Tuple[Optional[str], str]:
    vals = arg.split("=", 1)
    name: Optional[str]
    if len(vals) > 1:
        # Remove whitespace around the name
        name = vals[0].strip()
        # Expand the ~ to the full path as it won't be done automatically since it's
        # not at the beginning of the word.
        value = os.path.expanduser(vals[1])
    else:
        name = None
        value = vals[0]
    return name, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "preds",
        nargs="+",
        metavar="[NAME=]PATH",
        type=str,
        help=(
            "Path(s) to the prediction TSV file(s). "
            "If no name is specified it uses the name of the ground truth file."
        ),
    )
    parser.add_argument(
        "-g",
        "--gt",
        dest="gt",
        nargs="+",
        metavar="[NAME=]PATH",
        required=True,
        type=str,
        help=(
            "Path(s) to the ground truth TSV file(s). "
            "If no name is specified it uses the name of the ground truth file."
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        type=Path,
        help="Path of the output directory to store the evaluation results",
    )
    options = parser.parse_args()

    # The reusults are stored in a flat dictionary with the file path as key, but the
    # value is a dictionary containing the actual text, the source of the ground truth
    # and the predictions assigned to that ground truth.
    # i.e. "some/path" = {
    #           "text": "ocr result",
    #           "source": "GT Name",
    #           "preds": {
    #               "name": { "text": "pred result", "cer": 0.4, "wer": 0.5 },
    #            }
    # }
    results = {}
    invalid = {}
    sources = dict(gt=[], preds=[])
    for gt in options.gt:
        gt_name, gt_path_str = split_named_arg(gt)
        gt_path = Path(gt_path_str)
        if gt_name is None:
            gt_name = gt_path.name
        sources["gt"].append(gt_name)
        with open(gt_path, "r", encoding="utf-8") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            for line in reader:
                results[line[0]] = dict(text=line[1], source=gt_name, preds={})

    for pred in options.preds:
        pred_name, pred_path_str = split_named_arg(pred)
        pred_path = Path(pred_path_str)
        if pred_name is None:
            pred_name = pred_path.name
        sources["preds"].append(pred_name)

        out_fd = None
        out_tsv = None
        if options.out_dir:
            out_dir_pred = options.out_dir / pred_name
            out_dir_pred.mkdir(parents=True, exist_ok=True)
            out_fd = open(
                out_dir_pred / "individual_metrics.tsv", "w", encoding="utf-8"
            )
            out_tsv = csv.writer(
                out_fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            # Header of the individual metrics TSV
            out_tsv.writerow(["path", "pred", "gt", "cer", "wer", "correct"])

        spinner = Halo(text=f"Evaluating - {pred_name}")
        spinner.start()
        invalid[pred_name] = []
        with open(pred_path, "r", encoding="utf-8") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            for line in reader:
                key = line[0]
                pred_text = line[1]
                gt_text = results.get(key, {}).get("text")
                # Prediction of a file that is not part of the ground truth
                if gt_text is None:
                    invalid[pred_name].append(key)
                    spinner.stop()
                    print(
                        f" {pred_name} -- No ground truth for {key} available",
                        file=sys.stderr,
                    )
                    spinner.start()
                    if out_tsv is not None:
                        out_tsv.writerow([key, pred_text, "<UNKNOWN>", -1, -1])
                else:
                    cer = metrics.char_error_rate(pred_text, gt_text).item()
                    pred_result = dict(
                        text=pred_text,
                        cer=cer,
                        wer=metrics.word_error_rate(pred_text, gt_text).item(),
                        correct=cer == 0.0,
                    )
                    results[key]["preds"][pred_name] = pred_result

                    if out_tsv is not None:
                        out_tsv.writerow(
                            [
                                key,
                                pred_text,
                                gt_text,
                                pred_result["cer"],
                                pred_result["wer"],
                                pred_result["correct"],
                            ]
                        )
        spinner.succeed()
        if options.out_dir:
            out_fd.close()

    stats = summarise_results(
        results,
        invalid=invalid,
        gt_sources=sources["gt"],
        pred_sources=sources["preds"],
    )
    header = ["Group"]
    for gt_name in sources["gt"]:
        header.append(f"{gt_name} (CER)")
        header.append(f"{gt_name} (WER)")
        header.append(f"{gt_name} (Missing)")
        header.append(f"{gt_name} (Correct)")
    rows = []
    for pred_name, pred_stats in stats.items():
        row = [pred_name]
        for gt_name in sources["gt"]:
            row.append(pred_stats[gt_name]["cer"] * 100)
            row.append(pred_stats[gt_name]["wer"] * 100)
            row.append(pred_stats[gt_name]["missing"])
            row.append(pred_stats[gt_name]["correct"] * 100)
        rows.append(row)

    table = lavd.log.create_markdown_table(
        header=header,
        rows=rows,
        precision=2,
    )
    if options.out_dir:
        options.out_dir.mkdir(parents=True, exist_ok=True)
        with open(options.out_dir / "metrics.tsv", "w", encoding="utf-8") as out_fd:
            writer = csv.writer(
                out_fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            writer.writerow(header)
            writer.writerows(rows)
        with open(options.out_dir / "summary.md", "w", encoding="utf-8") as out_fd:
            out_fd.write("# Summary of Results")
            out_fd.write("\n")
            out_fd.write("\n")
            for line in table:
                out_fd.write(line)
                out_fd.write("\n")

    print("\n".join(table))


if __name__ == "__main__":
    main()
