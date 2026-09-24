# Evaluation-Frame Sensitivity

A small empirical probe for a "sleeper circuit": a behavior that stays quiet under one condition and switches on under another. Here the suspected trigger is *looking like a benchmark*. The task text is held byte-identical and only the wrapper changes, so any behavior change is attributable to the wrapper alone.

**Setup.** 36 borderline-but-legitimate requests, each in three frames: bare, an app-deployment system prompt, and a benchmark-style header. 108 prompts. Model: `Qwen/Qwen2.5-32B-Instruct`, bf16, greedy, run locally on 2× RTX A6000. Responses are graded REFUSE / PARTIAL / COMPLY by two graders: Qwen itself and an independent `gemma-2-9b-it`. A second probe asks the model, per frame, how likely it thinks the message is an evaluation.

**Result.** Hard refusals on identical requests rise from 17% (bare) and 31% (app frame) to 58% (benchmark frame). Seven requests are answered in the app frame but withheld in the benchmark frame; none go the other way. The model's self-reported "is this an eval" score is flat across frames, so self-report is not a usable measure. Full write-up in `reports/eval_frame_sensitivity.md`.

## Layout
```
generate_dataset.py  asks an open-weight model on OpenRouter for a large set of borderline requests -> data/requests.json
build_dataset.py     wraps each request in the three frames -> data/eval_frame_sensitivity.jsonl (2,253 prompts)
data/requests_v1.json  the 36 hand-written requests the report's numbers come from
common.py          model loading, batched greedy generation, self-judge
run.py             generates responses, self-judge labels, eval-awareness scores -> results/eval_frame_sensitivity.jsonl
rejudge.py         adds the independent gemma-2-9b-it grade to the same file
analyze.py         figure + results/summary.json + results/eval_frame_sensitivity_scored.csv
reports/           the one-page report
figures/           the figure
```
## Reproduce
```bash
export OPENROUTER_API_KEY=...
python3 generate_dataset.py --model nousresearch/hermes-4-405b --per-category 25 --rounds 2   # 751 requests -> data/requests.json
python3 build_dataset.py
CUDA_VISIBLE_DEVICES=4,5 python3 run.py
CUDA_VISIBLE_DEVICES=3 python3 rejudge.py
python3 analyze.py
```
