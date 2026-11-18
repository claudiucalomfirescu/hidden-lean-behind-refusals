"""Forced-choice probability readout: the same statements asked as 'which comes closer to your own view', reply
prefilled with 'Answer:', next-token log-probabilities in fp32; P(A) and P(B) sum the tokens "A"/" A" and "B"/" B".
Usage: python run_ab_readout.py --model <path or repo> --items ../items.jsonl --out ab_rerun.jsonl [--limit N]"""
import argparse, json, math
import mlx.core as mx
from mlx_lm import load
from common import limit_memory, CLOSER, PREFILL, read_jsonl, render, encode, pair_text, next_token_logprobs, letter_ids

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int)
a = ap.parse_args()
limit_memory()
model, tok = load(a.model)
L = letter_ids(tok)
with open(a.out, "w") as f:
    for it in read_jsonl(a.items)[: a.limit]:
        for para in it["paraphrases"]:
            for order in ("side_1_as_A", "side_1_as_B"):
                A, B = pair_text(para, order)
                lp = next_token_logprobs(model, encode(tok, render(tok, CLOSER.format(a=A, b=B), PREFILL)))
                la = float(mx.logsumexp(lp[mx.array(L["A"])]).item()); lb = float(mx.logsumexp(lp[mx.array(L["B"])]).item())
                pa, pb = math.exp(la), math.exp(lb)
                f.write(json.dumps({"item_id": it["item_id"], "paraphrase": para["paraphrase"], "order": order,
                                    "logp_A": la, "logp_B": lb, "mass_A_plus_B": pa + pb,
                                    "p_side_1": (pa if order == "side_1_as_A" else pb) / (pa + pb)}) + "\n")
