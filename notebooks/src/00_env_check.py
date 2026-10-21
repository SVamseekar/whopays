# whopays: standalone

# %% [markdown]
# # 00 · Environment check (Colab T4)
#
# Answers the PRD's week-1 open questions before any harness code depends on them. This
# notebook is **standalone**: it does not install `whopays`.
#
# **Run:** Runtime → Change runtime type → **T4 GPU**. Add `HF_TOKEN` in Secrets (key icon)
# and turn on notebook access; FLORES needs the token *and* the terms accepted at
# https://huggingface.co/datasets/facebook/flores. Then Runtime → **Run all** (~30–45 min;
# the GPTQModel install is the slow part).
#
# **Paste back:** the PASS/FAIL table and the JSON block printed by the last cell.
#
# GPTQModel and kvpress are installed into separate virtual environments that reuse the
# system torch, so their transformers requirements cannot change the main kernel.

# %%
PINNED_TRANSFORMERS = "5.19.0"  # must match requirements.lock

import importlib.metadata as md
import subprocess
import sys

torch_version = md.version("torch")
with open("/tmp/keep-torch.txt", "w") as f:
    f.write(f"torch=={torch_version}\n")
before = {p: md.version(p) for p in ("torch", "transformers")}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/keep-torch.txt",
                f"transformers=={PINNED_TRANSFORMERS}", "accelerate", "optimum-quanto", "hqq",
                "datasets"], check=True)
print("before:", before)
print("after: ", {p: md.version(p) for p in ("torch", "transformers", "optimum-quanto", "hqq")})

# %%
import json
import os
import textwrap
import time
import traceback

RESULTS: list[dict] = []
SMALL, TARGET = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
SIZES = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
         "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
GPTQ_CKPT = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


if (tok := secret("HF_TOKEN")):
    os.environ["HF_TOKEN"] = tok


def check(name):
    """Run fn; record PASS / FAIL / INFO with its returned detail and the elapsed time.

    fn returns (status, detail) or just detail (PASS). Exceptions become FAIL.
    """
    def deco(fn):
        start = time.time()
        try:
            out = fn()
            status, detail = out if isinstance(out, tuple) else ("PASS", out)
        except Exception as exc:
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        RESULTS.append({"check": name, "status": status, "detail": detail,
                        "seconds": round(time.time() - start, 1)})
        print(f"[{status}] {name}: {json.dumps(detail, ensure_ascii=False)[:400]}")
        return fn
    return deco


def run_isolated(name, packages, script, timeout):
    """pip install `packages` into a venv that sees system torch, then run `script` there."""
    venv = f"/tmp/venv-{name}"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv], check=True)
    pip = subprocess.run([f"{venv}/bin/pip", "install", "-q", "-c", "/tmp/keep-torch.txt",
                          *packages], capture_output=True, text=True, timeout=timeout)
    if pip.returncode != 0:
        raise RuntimeError("pip install failed: " + pip.stderr[-1500:])
    proc = subprocess.run([f"{venv}/bin/python", "-c", textwrap.dedent(script)],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1500:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


# %% [markdown]
# ## GPU and versions

# %%
import torch
import transformers


@check("gpu")
def _():
    assert torch.cuda.is_available(), "no CUDA device: switch the runtime to T4 GPU"
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "capability": f"{props.major}.{props.minor}",
            "vram_gb": round(props.total_memory / 2**30, 1), "count": torch.cuda.device_count()}


@check("versions")
def _():
    v = {"python": sys.version.split()[0], "torch": torch.__version__,
         "cuda": torch.version.cuda, "transformers": transformers.__version__,
         "accelerate": md.version("accelerate"), "optimum-quanto": md.version("optimum-quanto"),
         "hqq": md.version("hqq"), "datasets": md.version("datasets")}
    ok = transformers.__version__ == PINNED_TRANSFORMERS
    return ("PASS" if ok else "FAIL"), v


# %% [markdown]
# ## FP16 + SDPA baseline on Qwen2.5-0.5B

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

tok_small = AutoTokenizer.from_pretrained(SMALL)
small = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float16,
                                             attn_implementation="sdpa").cuda().eval()
PASSAGE = (" ".join(["The river town kept its records in a stone archive by the bridge."] * 30)
           + " Question: Where were the records kept? Answer:")
ctx = tok_small(PASSAGE, return_tensors="pt").input_ids.cuda()
question = tok_small(" In the stone archive by the", return_tensors="pt").input_ids.cuda()


@check("fp16_sdpa_generate")
def _():
    ids = tok_small("Write one sentence about rivers.", return_tensors="pt").input_ids.cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    out = small.generate(ids, max_new_tokens=64, do_sample=False, min_new_tokens=64)
    torch.cuda.synchronize()
    return {"tokens_per_s_bs1": round(64 / (time.time() - t0), 1),
            "text": tok_small.decode(out[0, ids.shape[1]:])[:80]}


@torch.no_grad()
def logits_after(cache):
    """Prefill PASSAGE into `cache`, then score `question` against it: last-position logits."""
    small(ctx, past_key_values=cache, use_cache=True)
    return small(question, past_key_values=cache, use_cache=True).logits[0, -1].float()


# %% [markdown]
# ## KV cache: quantized cache, manual eviction, crop (rollback for speculative decoding)

# %%
from transformers import DynamicCache


def make_quantized_cache(backend):
    """Try the transformers 5.x constructor, then the 4.x one; report which API worked."""
    try:
        from transformers import QuantizedCache
        return QuantizedCache(backend=backend, config=small.config, nbits=4), "QuantizedCache(5.x)"
    except (ImportError, TypeError):
        from transformers import QuantizedCacheConfig
        cls = {"quanto": "QuantoQuantizedCache", "hqq": "HQQQuantizedCache"}[backend]
        cfg = QuantizedCacheConfig(backend=backend, nbits=4)
        return getattr(transformers, cls)(cache_config=cfg), f"{cls}(4.x)"


full_logits = logits_after(DynamicCache())


@check("two_pass_vs_single_pass")
def _():
    # The KV protocol scores the question against a prefilled context cache. With the full
    # cache this must match one forward over context + question up to FP16 noise.
    with torch.no_grad():
        single = small(torch.cat([ctx, question], dim=1)).logits[0, -1].float()
    diff = (single - full_logits).abs().max().item()
    return ("PASS" if diff < 0.1 else "FAIL"), {
        "max_logit_diff": round(diff, 5), "argmax_same": bool(single.argmax() == full_logits.argmax())}

for backend in ("quanto", "hqq"):
    @check(f"quantized_cache_{backend}")
    def _(backend=backend):
        try:
            out = small.generate(question, max_new_tokens=16, do_sample=False,
                                 cache_implementation="quantized",
                                 cache_config={"backend": backend, "nbits": 4})
            generate = out.shape[1] > question.shape[1]
        except Exception as exc:
            generate = f"{type(exc).__name__}: {str(exc)[:200]}"
        cache, api = make_quantized_cache(backend)
        q_logits = logits_after(cache)
        # Belebele scores in one pass; this shows whether a prefilled-then-queried quantized
        # cache actually perturbs the answer logits (it must, or the KV factor is a no-op).
        diff = (q_logits - full_logits).abs().max().item()
        # Rollback for speculative decoding. QuantizedLayer inherits DynamicLayer.crop, which
        # only trims the FP16 residual buffer; here the question still sits in that buffer.
        cache.crop(-question.shape[1])
        with torch.no_grad():
            again = small(question, past_key_values=cache, use_cache=True).logits[0, -1].float()
        crop_diff = (again - q_logits).abs().max().item()
        return ("PASS" if diff > 0 else "FAIL"), {
            "api": api, "generate_ok": generate,
            "prefill_then_query_max_logit_diff": round(diff, 4),
            "argmax_same": bool(q_logits.argmax() == full_logits.argmax()),
            "crop_within_residual_max_diff": round(crop_diff, 5),
            "prompt_tokens": ctx.shape[1]}


def evicted(budget, sink=4):
    """Prefill, then keep `sink` first + most recent positions so the cache holds `budget`."""
    cache = DynamicCache()
    with torch.no_grad():
        small(ctx, past_key_values=cache, use_cache=True)
    n = ctx.shape[1]
    keep = torch.cat([torch.arange(min(sink, budget)),
                      torch.arange(max(sink, n - (budget - sink)), n)]).cuda()
    keep = keep[keep < n].unique()
    layers = getattr(cache, "layers", None)
    for i in range(len(layers) if layers is not None else len(cache.key_cache)):
        if layers is not None:
            layers[i].keys = layers[i].keys[:, :, keep]
            layers[i].values = layers[i].values[:, :, keep]
        else:
            cache.key_cache[i] = cache.key_cache[i][:, :, keep]
            cache.value_cache[i] = cache.value_cache[i][:, :, keep]
    pos = torch.arange(n, n + question.shape[1], device="cuda")[None]
    with torch.no_grad():
        out = small(question, past_key_values=cache, position_ids=pos, use_cache=True)
    return out.logits[0, -1].float(), cache.get_seq_length()


@check("manual_kv_eviction")
def _():
    same, kept_all = evicted(budget=10_000)
    diff_noop = (same - full_logits).abs().max().item()
    cut, kept = evicted(budget=256)
    ok = diff_noop < 1e-2
    return ("PASS" if ok else "FAIL"), {
        "no_eviction_max_diff": round(diff_noop, 5), "kept_tokens_after_cut": kept - question.shape[1],
        "cut_max_logit_diff": round((cut - full_logits).abs().max().item(), 3),
        "prompt_tokens": ctx.shape[1]}


@check("dynamic_cache_crop")
def _():
    cache = DynamicCache()
    with torch.no_grad():
        small(ctx, past_key_values=cache, use_cache=True)
        n = cache.get_seq_length()
        small(question, past_key_values=cache, use_cache=True)
        cache.crop(-question.shape[1])  # 5.x API: negative = tokens to remove
        again = small(question, past_key_values=cache, use_cache=True).logits[0, -1].float()
    return {"length_after_crop": n, "max_diff_vs_uncropped": round(
        (again - full_logits).abs().max().item(), 5)}


# %% [markdown]
# ## Tokenizers: are draft (0.5B) and target (1.5B / 3B / 7B) token IDs identical?

# %%
SENTENCES = [
    "The weather is nice today, so we will walk to the market.",
    "Mary bought 3 apples for $1.25 each. How much did she pay?",
    "Das Wetter ist heute schön, also gehen wir zum Markt.",
    "Le chat dort sur le canapé depuis ce matin.",
    "El tren sale a las ocho y media de la mañana.",
    "Сегодня хорошая погода, поэтому мы пойдём на рынок.",
    "今天天气很好，我们走路去市场吧。",
    "今日はいい天気なので、市場まで歩いて行きましょう。",
    "วันนี้อากาศดี เราจะเดินไปตลาดกัน",
    "আজ আবহাওয়া ভালো, তাই আমরা বাজারে হেঁটে যাব।",
    "Hali ya hewa ni nzuri leo, kwa hiyo tutatembea hadi sokoni.",
    "ఈ రోజు వాతావరణం బాగుంది, కాబట్టి మనం మార్కెట్‌కు నడిచి వెళ్దాం.",
    "Answer: (C) 42 — see https://example.org/q?id=7 😊",
    "Wie viele Äpfel hat Anna, wenn sie 12 hat und 5 verschenkt?",
    "Combien font 17 fois 23 ?",
    "Сколько будет 144 разделить на 12?",
    "小明有15个苹果，他吃了4个，还剩几个？",
    "เด็กชายมีลูกแก้ว 25 ลูก เขาให้เพื่อนไป 7 ลูก",
    "রিনার কাছে ১২টি কলম আছে।",
    "రాము దగ్గర 20 రూపాయలు ఉన్నాయి.",
]


@check("tokenizer_alignment")
def _():
    from transformers import AutoConfig

    ref = AutoTokenizer.from_pretrained(SIZES[0])
    ref_ids = [ref(s).input_ids for s in SENTENCES]
    detail = {}
    for name in SIZES:
        t = AutoTokenizer.from_pretrained(name)
        mism = sum(t(s).input_ids != r for s, r in zip(SENTENCES, ref_ids))
        detail[name.split("/")[-1]] = {"len_tokenizer": len(t),
                                       "embedding_rows": AutoConfig.from_pretrained(name).vocab_size,
                                       "sentences_with_different_ids": mism}
    ok = all(v["sentences_with_different_ids"] == 0 for v in detail.values())
    return ("PASS" if ok else "FAIL"), detail


# %% [markdown]
# ## FP16 speculative decoding vs plain greedy (1.5B target, 0.5B draft)
#
# Under FP16 the verify pass runs k+1 tokens at once while plain greedy runs one at a time, so
# rounding can differ and flip near-tied argmaxes. This measures how often that happens, and
# the top-2 logit margin at the first divergence, to set the FR-5 exactness criterion.

# %%
@check("fp16_assisted_vs_greedy")
def _():
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float16,
                                                  attn_implementation="sdpa").cuda().eval()
    prompts = SENTENCES[2:12]
    detail = {"prompts": len(prompts), "identical": 0, "divergences": []}
    for p in prompts:
        msgs = [{"role": "user", "content": p + " Continue the text."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).cuda()
        plain = target.generate(ids, max_new_tokens=64, do_sample=False,
                                output_scores=True, return_dict_in_generate=True)
        spec = target.generate(ids, max_new_tokens=64, do_sample=False, assistant_model=small,
                               num_assistant_tokens=4, num_assistant_tokens_schedule="constant")
        a, b = plain.sequences[0, ids.shape[1]:], spec[0, ids.shape[1]:]
        n = min(len(a), len(b))
        diff = (a[:n] != b[:n]).nonzero()
        if len(diff) == 0 and len(a) == len(b):
            detail["identical"] += 1
            continue
        pos = int(diff[0]) if len(diff) else n
        top2 = plain.scores[min(pos, len(plain.scores) - 1)][0].float().topk(2).values
        detail["divergences"].append({"first_diff_at": pos,
                                      "greedy_top2_margin": round(float(top2[0] - top2[1]), 4)})
    del target
    torch.cuda.empty_cache()
    return "INFO", detail


# %% [markdown]
# ## Benchmarks reachable (pinned revisions are recorded in the result)

# %%
@check("datasets")
def _():
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    detail = {}
    for repo, path in [("facebook/belebele", "data/tel_Telu.jsonl"),
                       ("juletxara/mgsm", "te/test-00000-of-00001.parquet"),
                       ("facebook/flores", "data/language/tel_Telu/devtest-00000-of-00001.parquet")]:
        try:
            sha = api.dataset_info(repo).sha
            hf_hub_download(repo, path, repo_type="dataset", revision=sha)
            detail[repo] = {"ok": True, "sha": sha}
        except Exception as exc:
            detail[repo] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    ok = all(v["ok"] for v in detail.values())
    return ("PASS" if ok else "FAIL"), detail


small = None  # free VRAM before the isolated-venv checks
torch.cuda.empty_cache()

# %% [markdown]
# ## kvpress (isolated venv; it requires an older transformers)

# %%
@check("kvpress")
def _():
    return run_isolated("kvpress", ["kvpress"], f"""
        import json, torch, transformers, importlib.metadata as md
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from kvpress import KnormPress
        tok = AutoTokenizer.from_pretrained("{SMALL}")
        m = AutoModelForCausalLM.from_pretrained("{SMALL}", dtype=torch.float16).cuda().eval()
        ctx = tok({PASSAGE!r}, return_tensors="pt").input_ids.cuda()
        q = tok(" In the stone archive by the", return_tensors="pt").input_ids.cuda()
        press = KnormPress(compression_ratio=0.5)
        with torch.no_grad(), press(m):
            out = m(ctx, use_cache=True)
        kept = out.past_key_values.get_seq_length()
        with torch.no_grad():
            nxt = m(q, past_key_values=out.past_key_values).logits[0, -1].argmax().item()
        print(json.dumps({{"kvpress": md.version("kvpress"), "transformers": transformers.__version__,
                          "prompt_tokens": ctx.shape[1], "kept_after_press": kept,
                          "next_token": tok.decode(nxt)}}))
    """, timeout=1200)


# %% [markdown]
# ## GPTQModel (isolated venv; only needed as the FR-3 reference)

# %%
@check("gptqmodel")
def _():
    return run_isolated("gptq", ["gptqmodel"], f"""
        import json, time, torch, importlib.metadata as md
        from gptqmodel import GPTQModel
        t0 = time.time()
        m = GPTQModel.load("{GPTQ_CKPT}")
        tok = m.tokenizer
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(m.device)
        out = m.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
        qlinear = next((type(x).__name__ for x in m.model.modules()
                        if "QuantLinear" in type(x).__name__), "none found")
        print(json.dumps({{"gptqmodel": md.version("gptqmodel"), "kernel": qlinear,
                          "load_s": round(time.time() - t0, 1),
                          "text": tok.decode(out[0, ids.shape[1]:])}}))
    """, timeout=2400)


# %% [markdown]
# ## Summary: paste everything below back to Claude Code

# %%
width = max(len(r["check"]) for r in RESULTS)
print(f"{'check':<{width}}  status  seconds")
for r in RESULTS:
    print(f"{r['check']:<{width}}  {r['status']:<6}  {r['seconds']:>7}")
print("\n" + json.dumps(RESULTS, ensure_ascii=False, indent=1))
