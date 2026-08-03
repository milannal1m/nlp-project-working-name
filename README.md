# Abstractive News Summarization by QLoRA Fine-Tuning of Phi-3-mini

QLoRA fine-tuning of `microsoft/Phi-3-mini-4k-instruct` for abstractive
summarization of CNN/DailyMail and XSum. A LoRA adapter is trained on top of a
4-bit NF4-quantized base; at inference the same 4-bit base is loaded and the
adapter attached, so the fine-tuned and original configurations differ only by
the adapter.

The trained adapter is included (`adapters/phi3-lora-news-full`, 96 MB,
safetensors). Inference requires a GPU; training was performed on 4× A100.

---

## Method and pipeline

**Base and adaptation.** `microsoft/Phi-3-mini-4k-instruct` loaded in 4-bit NF4
with double quantization and bfloat16 compute. LoRA with r=16, α=32, dropout
0.05, bias none, applied to `qkv_proj`, `o_proj`, `gate_up_proj` and
`down_proj` — Phi-3 uses fused projections, so the Llama-style
`q_proj`/`k_proj`/`v_proj` names do not match. 25,165,824 trainable parameters,
0.65% of 3,846,245,376.

**Objective.** Completion-only loss: prompt tokens are masked to `-100` and the
loss is taken on the summary plus a terminating EOS, so the model learns where
to stop rather than running to the token limit.

**Prompt.** Identical string at training and inference, taken from a single
source of truth in `model.py`, so no train/test format skew is possible.

**Sequence length.** 2048 tokens, matched to the inference
`max_input_length`. 35.9% of training examples exceed 1024 tokens, so a shorter
budget truncates a third of the corpus at training time only.

**Data.** Full train splits, unbalanced: 287,113 CNN/DailyMail + 204,017 XSum =
491,130 pairs, tokenized once and cached before training.

**Optimization.** Effective batch 32 (per-device 4 × grad-accum 2 × 4 GPUs),
15,348 optimizer steps, 1 epoch, `adamw_torch`, lr 2e-4 cosine with 3% warmup,
max-grad-norm 0.3, gradient checkpointing, bf16. Checkpoints every 250 steps
with automatic resume.

**Generation.** Greedy decoding, `no_repeat_ngram_size=3`, `min_new_tokens=8`,
and per-dataset budgets `max_new_tokens` = 128 for CNN/DailyMail and 64 for
XSum. Outputs are post-processed to strip instruction preambles and to drop a
trailing incomplete sentence when the token cap was reached.

```bash
pip install -r requirements.txt

python train_lora.py --prepare_only --per_dataset_samples 0 \
    --max_seq_length 2048 --tokenized_path data/tokenized_full

torchrun --nproc_per_node=4 train_lora.py \
    --tokenized_path data/tokenized_full --max_seq_length 2048 \
    --output_dir adapters/phi3-lora-news-full --batch_size 4 --grad_accum 2

python run_generation.py --index 11 --dataset xsum
python run_evaluation.py --index 23
python aggregate.py
```

`--index 11` is `Phi-3-LoRA-Full_4bit`; indices 7–9 are the original Phi-3 at
fp16 / 4-bit / 8-bit. SLURM job scripts for each stage are under `slurm/`.

---

## Results

Full test splits, no sampling: 11,490 CNN/DailyMail and 11,334 XSum documents.

The controlled comparison is **fine-tuned vs original at the same
quantization** — `Phi-3-LoRA-Full_4bit` against `Phi-3_4bit`. Both load the same
base checkpoint in 4-bit NF4; the adapter is the only difference.

### XSum

| | original fp16 | original 8-bit | original 4-bit | fine-tuned 4-bit | Δ vs original 4-bit |
|---|---|---|---|---|---|
| ROUGE-L | 0.1131 | 0.1136 | 0.1189 | **0.2969** | +0.1781 (+150%) |
| BLEU | 0.0121 | 0.0124 | 0.0122 | **0.0911** | +0.0789 (+644%) |
| METEOR | 0.2257 | 0.2267 | 0.2261 | **0.3222** | +0.0961 (+42%) |
| BERTScore-F1 | 0.8527 | 0.8526 | 0.8537 | **0.9010** | +0.0473 |
| words | 102.9 | 103.3 | 97.3 | 20.3 | reference 21.1 |
| sentences | 3.82 | 3.91 | 3.97 | 1.04 | reference 1.0 |

### CNN/DailyMail

| | original fp16 | original 8-bit | original 4-bit | fine-tuned 4-bit | Δ vs original 4-bit |
|---|---|---|---|---|---|
| ROUGE-L | 0.1846 | 0.1854 | 0.1836 | **0.2222** | +0.0386 (+21%) |
| BLEU | 0.0485 | 0.0500 | 0.0455 | **0.0503** | +0.0048 (+10%) |
| METEOR | 0.3008 | 0.3035 | 0.2880 | 0.2767 | −0.0113 (−4%) |
| BERTScore-F1 | 0.8521 | 0.8521 | 0.8510 | **0.8706** | +0.0196 |
| words | 103.0 | 103.9 | 98.4 | 51.2 | reference 52.9 |
| sentences | 3.78 | 3.83 | 3.94 | 3.99 | reference 3.4 |

Reference points on the same splits: Lead-3 scores ROUGE-L 0.2428 on
CNN/DailyMail and 0.1155 on XSum.

![rougeL](results/charts/rougeL.png)
![meteor](results/charts/meteor.png)
![bleu](results/charts/bleu.png)
![bertscore](results/charts/bertscore_f1.png)

### Training configuration

| | |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct`, 4-bit NF4, double quant, bf16 compute |
| LoRA | r=16, α=32, dropout 0.05, bias none |
| Target modules | `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` |
| Trainable parameters | 25,165,824 / 3,846,245,376 (0.65%) |
| Training pairs | 491,130 (287,113 CNN/DailyMail + 204,017 XSum) |
| Sequence length | 2048 |
| Effective batch | 32 (4 × 2 × 4 GPUs) |
| Steps / epochs | 15,348 / 1 |
| Optimizer | `adamw_torch`, lr 2e-4, cosine, warmup 3%, max-grad-norm 0.3 |
| Final training loss | 1.1096 |
| Hardware | 4 × NVIDIA A100 80GB, DDP |
| Wall-clock | 27.1 h training; 15.4 h + 6.0 h generation over the two test splits |

---

## Generated summaries against references

Three cases from the fine-tuned model.

### 1. XSum

**Article.** Ben Gerring, 29, was in the water near Mandurah, south of Perth,
when the attack occurred on Tuesday afternoon. Western Australia Police said Mr
Gerring's injuries were too severe to overcome and he died on Friday night. On
Thursday, a 4.2m (14ft) great white shark was captured near Mandurah, but it is
not known if this shark was to blame for the attack. "He was right out the back
looking to get a big wave, one of the set waves that had been coming through
during the day," Brian Williams, president of the Mandurah Boardriders Club and
friend of the victim, told ABC on Tuesday. "He'd sort of paddled out the back,
sort of past the pack slightly. And next thing all hell broke loose and they
were trying to bring him in. "His board was broken in half, they found the tail
part of his board," Mr Williams said. Surf Lifesaving Western Australia had
reportedly issued a warning before the attack, after an unknown species of shark
was seen in the area.

**Reference.** A surfer who lost his leg in a shark attack in Western Australia
last week has died, police say.

**Generated.** A surfer has died after being attacked by a shark in Western
Australia.

### 2. XSum

**Article.** Former Exeter player Pope, 26, can also play on the wing and joined
the Championship club in 2013. May, 22, has made 69 appearances for the
Penzance-based side, and has been their top point-scorer for the past two
seasons. Moyle, 24, can play at full-back, wing or fly-half, and made his
Pirates debut in 2012. "All three individuals have developed as players, and
Laurence is now one of the standout 10s in the Championship," said coach Gavin
Cattle.

**Reference.** Scrum-half Mike Pope, fly-half Laurence May and back Kyle Moyle
have all signed new one-year deals at Cornish Pirates.

**Generated.** Cornish Pirates have signed England Under-20s back-row forward
Laurence May, prop Luke Pope and centre Jamie Moyle on two-year contracts.

### 3. CNN/DailyMail

**Article.** What do we have for the contestant on "The Price Is Right"? A
brand-new car! Whoops. That wasn't supposed to happen. On Thursday's edition of
the popular game show, model Manuela Arbelaez accidentally revealed the correct
answer to a guessing game for a new Hyundai Sonata. Host Drew Carey couldn't
stop laughing. "Congratulations! Manuela just gave you a car!" he exulted.
Arbelaez was mortified, attempting to hide behind the display. But everything
turned out OK, she tweeted later. It's been a busy week for "The Price Is
Right." On Wednesday, former host Bob Barker, 91, showed up to run his old show.

**Reference.** "The Price Is Right" gives away a car ... accidentally .
A model makes a big mistake during a game .
Host Drew Carey thought the error was hilarious .

**Generated.** Manuela accidentally reveals the answer to the guessing game .
The model is mortified .
Bob Barkers shows up to host his old game show .
