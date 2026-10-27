"""Tests for the `reasoning_history` option of hf/ver4.0/alpha_2.1/chat_template.jinja.

    python tests/test_reasoning_history.py [--baseline <original template>]

Needs `transformers` (any 4.x/5.x with apply_chat_template). The template is rendered as text
(tokenize=False) through a plain tokenizers-backed PreTrainedTokenizerFast, so the repo's custom
tokenizer class is not involved. --baseline is the template before this option existed
(default: hf/ver4.0/alpha_2.0/chat_template.jinja).
"""
import argparse
import os
import sys

from transformers import PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "hf/ver4.0/alpha_2.1/chat_template.jinja")
BASELINE = os.path.join(ROOT, "hf/ver4.0/alpha_2.0/chat_template.jinja")
TOKENIZER_JSON = os.path.join(ROOT, "hf/ver4.0/alpha_2.1/tokenizer.json")
KW = dict(model_identity="You are LLM-jp-4.", reasoning_effort="medium", knowledge_cutoff="2025-12",
          conversation_start_date="2026-10-20")
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a command",
                                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                                           "required": ["command"]}}}]


def call(cid, cmd):
    return {"id": cid, "type": "function", "function": {"name": "bash", "arguments": '{"command": "%s"}' % cmd}}


def conversation(field="thinking"):
    """user1 -> A1 call (T1) -> R1 -> A2 call (T2) -> R2 -> A3 final (T3) -> user2 -> A4 call (T4) -> R4 -> A5 final (T5)."""
    def a(reasoning, **rest):
        m = {"role": "assistant", **rest}
        m[field] = reasoning
        return m
    return [
        {"role": "developer", "content": "Be helpful."},
        {"role": "user", "content": "first question"},
        a("T1-reasoning", content="", tool_calls=[call("c1", "ls")]) if field != "thinking" else a("T1-reasoning", tool_calls=[call("c1", "ls")]),
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "R1-output"},
        a("T2-reasoning", tool_calls=[call("c2", "cat a")]),
        {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "R2-output"},
        a("T3-reasoning", content="F1-answer"),
        {"role": "user", "content": "second question"},
        a("T4-reasoning", tool_calls=[call("c4", "pwd")]),
        {"role": "tool", "tool_call_id": "c4", "name": "bash", "content": "R4-output"},
        a("T5-reasoning", content="F5-answer"),
    ]


def make(template):
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON)
    tok.chat_template = template
    def render(messages, gen, **extra):
        return tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=gen, **KW, **extra)
    return render


def shown(text):
    return [t for t in ("T1", "T2", "T3", "T4", "T5") if f"{t}-reasoning" in text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    new = make(open(TEMPLATE, encoding="utf-8").read())
    base_src = open(args.baseline or BASELINE, encoding="utf-8").read()
    old = make(base_src)
    failures = []

    def check(name, got, want):
        ok = got == want
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"\n     got:  {got}\n     want: {want}"))
        if not ok:
            failures.append(name)

    conv = conversation()
    inference_prefix = conv[:10]          # ... R4, next turn to be generated
    training_full = conv                  # target = A5 final (last message)
    # 1. which earlier reasoning is visible at inference (add_generation_prompt=True)
    for mode, want in [("none", []), ("toolloop", ["T4"]), ("keepall", ["T1", "T2", "T3", "T4"])]:
        check(f"inference visibility, {mode}", shown(new(inference_prefix, True, reasoning_history=mode)), want)
    # 2. training target render (last message is the target): its own reasoning is always shown
    for mode, want in [("none", ["T5"]), ("toolloop", ["T4", "T5"]), ("keepall", ["T1", "T2", "T3", "T4", "T5"])]:
        check(f"training target visibility, {mode}", shown(new(training_full, False, reasoning_history=mode)), want)
    # 3. OpenAI-compatible reasoning fields (what vLLM passes) are honoured like `thinking`
    for field in ("reasoning_content", "reasoning"):
        c = conversation(field)
        check(f"field {field}, toolloop", shown(new(c[:10], True, reasoning_history="toolloop")), ["T4"])
        check(f"field {field}, keepall", shown(new(c[:10], True, reasoning_history="keepall")), ["T1", "T2", "T3", "T4"])
        check(f"field {field}, none", shown(new(c[:10], True, reasoning_history="none")), [])
    # 4. none / default are byte-identical to the original template
    for gen, msgs, label in [(True, inference_prefix, "inference"), (False, training_full, "training")]:
        ref = old(msgs, gen)
        check(f"default == original ({label})", new(msgs, gen) == ref, True)
        check(f"none == original ({label})", new(msgs, gen, reasoning_history="none") == ref, True)
    for k in range(2, len(conv) + 1):     # every prefix, both prompt kinds
        for gen in (True, False):
            if conv[k - 1]["role"] != "assistant" and not gen:
                continue
            if new(conv[:k], gen) != old(conv[:k], gen):
                failures.append(f"default != original at prefix {k} gen={gen}")
    print("PASS default == original on every prefix" if not any("prefix" in f for f in failures) else "FAIL default differs on some prefix")
    # 5. prefix property for training: prompt render of turn k is a prefix of its full render
    for mode in ("none", "toolloop", "keepall"):
        bad = []
        for i, m in enumerate(conv):
            if m["role"] != "assistant":
                continue
            prompt = new(conv[:i], True, reasoning_history=mode)
            full = new(conv[:i + 1], False, reasoning_history=mode)
            if not full.startswith(prompt):
                bad.append(i)
        check(f"prompt is a prefix of the full render, every target, {mode}", bad, [])
    # 6. invalid values are rejected
    try:
        new(inference_prefix, True, reasoning_history="all")
        check("invalid value rejected", False, True)
    except Exception as error:  # noqa: BLE001 - jinja raises TemplateError subclasses
        check("invalid value rejected", "reasoning_history must be one of" in str(error), True)
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
