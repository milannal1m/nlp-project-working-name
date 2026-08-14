import argparse

import json

import os

import statistics


from pipeline_config import EVAL_TARGETS, OUTPUT_DIR, METRICS_DIR, SAMPLE

from evaluator import Evaluator, tokenize, sent_count


def _truncate(src: str, dst: str, sample: int | None) -> int:


    os.makedirs(os.path.dirname(dst), exist_ok=True)

    n = 0

    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:

        for line in fin:

            if sample is not None and n >= sample:

                break

            fout.write(line)

            n += 1

    return n


def _length_stats(path: str) -> dict:


    news_lens, ref_lens, gen_lens, gen_sents, compression = [], [], [], [], []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            rec = json.loads(line)

            news_tok = tokenize(rec.get("news", ""))

            gen_tok = tokenize(rec.get("generated_summary", ""))

            news_lens.append(len(news_tok))

            ref_lens.append(len(tokenize(rec.get("reference_summary", ""))))

            gen_lens.append(len(gen_tok))

            gen_sents.append(sent_count(rec.get("generated_summary", "")))

            if news_tok:

                compression.append(len(gen_tok) / len(news_tok))


    mean = lambda xs: statistics.mean(xs) if xs else 0.0

    std = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0

    return {

        "avg_news_len": mean(news_lens),

        "avg_ref_len": mean(ref_lens),

        "avg_gen_len": mean(gen_lens),

        "std_gen_len": std(gen_lens),

        "avg_gen_sents": mean(gen_sents),

        "avg_compression": mean(compression),

    }


def evaluate_target(target: dict, sample: int) -> dict:


    src = os.path.join(OUTPUT_DIR, target["filename"])

    result = {

        "label": target["label"],

        "dataset": target["dataset"],

        "file": target["filename"],

        "num_samples": 0,

        "errors": {},

    }


    if not os.path.exists(src):

        result["errors"]["file"] = f"missing summary file: {src}"

        print(f"[warn] {src} not found — writing empty metrics", flush=True)

        return result


    eval_input = os.path.join("results", "eval_inputs", target["filename"])

    n = _truncate(src, eval_input, sample)

    result["num_samples"] = n

    print(f"Evaluating {target['filename']} on {n} samples", flush=True)


    evaluator = Evaluator()


    try:

        result.update(evaluator.evaluate_metrics(eval_input))

        print("  [ok] BLEU / ROUGE-L / METEOR / BERTScore", flush=True)

    except Exception as e:

        result["errors"]["overlap"] = repr(e)

        print(f"  [fail] overlap metrics: {e}", flush=True)


    try:

        result.update(evaluator.evaluate_summac(eval_input))

        print("  [ok] SummaC", flush=True)

    except Exception as e:

        result["errors"]["summac"] = repr(e)

        print(f"  [fail] SummaC: {e}", flush=True)


    try:

        qa = evaluator.evaluate_qa(eval_input)

        if qa.get("error"):

            result["errors"]["qa_eval"] = qa["error"]

        result.update({k: v for k, v in qa.items() if k != "error"})

        print("  [ok] QAFactEval", flush=True)

    except Exception as e:

        result["errors"]["qa_eval"] = repr(e)

        print(f"  [fail] QAFactEval: {e}", flush=True)


    try:

        result.update(_length_stats(eval_input))

    except Exception as e:

        result["errors"]["length"] = repr(e)


    return result


def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate one summary file")

    parser.add_argument(

        "--index",

        type=int,

        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)),

        help="Index into pipeline_config.EVAL_TARGETS (defaults to $SLURM_ARRAY_TASK_ID)",

    )

    parser.add_argument("--sample", type=int, default=SAMPLE)

    parser.add_argument("--metrics_dir", type=str, default=METRICS_DIR)

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if not (0 <= args.index < len(EVAL_TARGETS)):

        raise SystemExit(

            f"--index {args.index} out of range (0..{len(EVAL_TARGETS) - 1})."

        )


    target = EVAL_TARGETS[args.index]

    os.makedirs(args.metrics_dir, exist_ok=True)

    result = evaluate_target(target, args.sample)


    out_path = os.path.join(args.metrics_dir, f"{target['label']}__{target['dataset']}.json")

    with open(out_path, "w", encoding="utf-8") as f:

        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":

    main()
