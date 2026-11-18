"""Written replies to the 'Personally, which do you support?' prompt: every item x paraphrase x letter order,
greedy decoding, max_tokens new tokens, one prompt at a time. Writes JSONL.
Usage: python run_replies.py --model <path or repo> --items ../items.jsonl --out replies_rerun.jsonl [--max-tokens 30] [--limit N]"""
import argparse, json, time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from common import limit_memory, PERSONAL, read_jsonl, render, encode, pair_text, label

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=30); ap.add_argument("--limit", type=int)
a = ap.parse_args()
limit_memory()
model, tok = load(a.model)
items = read_jsonl(a.items)[: a.limit]
t0 = time.time()
with open(a.out, "w") as f:
    for it in items:
        for para in it["paraphrases"]:
            for order in ("side_1_as_A", "side_1_as_B"):
                A, B = pair_text(para, order)
                prompt = PERSONAL.format(a=A, b=B)
                reply, last = "", None
                for r in stream_generate(model, tok, encode(tok, render(tok, prompt)), max_tokens=a.max_tokens,
                                         sampler=make_sampler(temp=0.0)):
                    reply += r.text; last = r
                lab = label(reply)
                side = None
                if lab.startswith("answered"):
                    side = "side_1" if (lab[-1] == "A") == (order == "side_1_as_A") else "side_2"
                f.write(json.dumps({"item_id": it["item_id"], "topic": it["topic"], "paraphrase": para["paraphrase"],
                                    "order": order, "prompt": prompt, "reply": reply, "label": lab, "answered_side": side,
                                    "generated_tokens": last.generation_tokens, "finish_reason": last.finish_reason},
                                   ensure_ascii=False) + "\n")
print(f"done in {time.time() - t0:.0f} s")
