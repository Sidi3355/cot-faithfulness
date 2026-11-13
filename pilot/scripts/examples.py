"""Writes pilot/results/EXAMPLES.md: real model outputs for a few questions.

Usage: python pilot/scripts/examples.py
Selection is fixed, not chosen by hand: the first N_QUESTIONS questions in the
evaluation order; for each, the first inserted word (in the frozen file's order)
whose first sampled trace (uncapped) changed the answer, and the first that did
not. Every text is copied from pilot/results/raw/ unchanged.
"""

import pandas as pd

import analyze

N_QUESTIONS = 3
SPEC = analyze.SPEC
EXPLAINERS = (  # (config, heading)
    ("C1_pe_with_explanations_empty", "Predict then explain, examples with explanations (C1, greedy)"),
    ("C1_pe_labels_only_empty", "Predict then explain, labels-only examples (C1, greedy)"),
    ("C3A_empty", "Explain then predict, labels-only examples (C3A, first of 3 samples)"),
)


def fence(text) -> str:
  text = "(no text)" if not isinstance(text, str) or not text.strip() else text.strip()
  return "```text\n" + text.replace("```", "'''") + "\n```\n"


def yes_no(x) -> str:
  return "–" if pd.isna(x) else ("yes" if x else "no")


def snippet(word: str, text) -> str:
  """The first place the word appears in the text, with a little context."""
  if not isinstance(text, str):
    return ""
  i = text.lower().find(word.lower())
  if i < 0:
    return ""
  start, end = max(0, i - 60), min(len(text), i + len(word) + 60)
  before, hit, after = text[start:i], text[i:i + len(word)], text[i + len(word):end]
  clean = lambda s: " ".join(s.split())
  return f' — first at: "…{clean(before)} **{clean(hit)}** {clean(after)}…"'


def sentences(row: pd.Series, word: str | None = None, key: str | None = None) -> str:
  lines = []
  for k in ("sent0", "sent1"):
    text = row[k]
    if word and k == key:
      text = text.replace(word, f"**{word}**", 1)
    lines.append(f"> SENTENCE {k[-1]}: {text}")
  return "  \n".join(lines) + "\n"


def one(rows: pd.DataFrame, example_id, row_type: str, intervention_idx=None, sample=0) -> pd.Series | None:
  sel = rows[(rows.example_id == example_id) & (rows.row_type == row_type) & (rows.sample_idx == sample)]
  if "budget" in sel and (sel.budget == "uncapped").any():
    sel = sel[sel.budget == "uncapped"]
  if intervention_idx is not None:
    sel = sel[sel.intervention_idx == intervention_idx]
  return None if sel.empty else sel.iloc[0]


def section_case(out, title, ex, trace_rows, expl_rows, data, iv=None):
  out.append(f"### {title}\n")
  if iv is not None:
    edited = data.ints[(data.ints.data_idx == ex) & (data.ints.intervention_idx == iv)].iloc[0]
    out.append(sentences(edited, edited["inserted_word"], edited["modified_sentence_key"]))
  row_type = "base" if iv is None else "real"
  t = one(trace_rows, ex, row_type, iv)
  out.append("**Thinking on (C2, uncapped, first of 3 samples)**\n")
  if t is None:
    out.append("(missing)\n")
  else:
    line = f"- Answer: {t['label'] if isinstance(t['label'], str) else 'unreadable'}"
    if iv is not None:
      line += (f"; changed from the original question's majority answer ({t['base_label']}): "
               f"{yes_no(t['I'])}; inserted word mentioned in the trace: {yes_no(t['E_paper'])}"
               + snippet(t["inserted_word"], t["trace"]))
    else:
      samples = trace_rows[(trace_rows.example_id == ex) & (trace_rows.row_type == "base")
                           & (trace_rows.budget == "uncapped")].sort_values("sample_idx")
      line += (f" (all 3 samples: {', '.join(str(l) for l in samples['label'])}; majority: {t['base_label']}; "
               f"correct answer: {t['gold_label']})")
    out.append(line + f"\n- Trace length: {int(t['trace_n_tokens'])} tokens\n")
    out.append(f"<details><summary>Full thinking trace</summary>\n\n{fence(t['trace'])}\n</details>\n")
    out.append(f"<details><summary>Text after the thinking</summary>\n\n{fence(t['answer_text'])}\n</details>\n")
  out.append("\n**Thinking off: written explanations (no length instruction)**\n")
  for (name, heading), rows in zip(EXPLAINERS, expl_rows):
    e = one(rows, ex, row_type, iv)
    if e is None:
      out.append(f"- *{heading}*: (missing)\n")
      continue
    label = e["label"] if isinstance(e["label"], str) else "unreadable"
    line = f"- *{heading}*: answer {label}"
    if iv is not None:
      line += (f"; changed: {yes_no(e['I'])}; inserted word mentioned: {yes_no(e['E_paper'])}")
    out.append(line + "\n\n" + fence(e["answer_text"]))
  out.append("")


def main() -> None:
  data = analyze.Data()
  traces = data.load(analyze.cfg_named("C2_thinking"))
  expl = [data.load(analyze.cfg_named(name)) for name, _ in EXPLAINERS]
  order = data.test.sample(frac=1, random_state=0).index
  out = [
      "# Example outputs\n",
      "Real outputs from the pilot, copied unchanged from `raw/`. Model: Qwen3-0.6B (arm C).\n",
      f"**How these were picked** (no hand-picking): the first {N_QUESTIONS} questions in the evaluation "
      "order. For each, the first inserted word whose trace changed the answer, and the first whose trace "
      "did not. \"Changed\" compares one answer with the majority of the 3 answers to the original "
      "question; \"mentioned\" uses the paper's rule (the word, or a word with the same stem, appears "
      "in the text).\n",
      "The budget-limited traces (128 to 1,024 tokens) are the same traces cut at that length, followed "
      "by Qwen's stop phrase and a fresh answer. The prompts are shown in [NOTES.md](NOTES.md) and the "
      "full prompt text is built by `src/cotfaith/prompts.py`.\n",
  ]
  for n, ex in enumerate([e for e in order if e in set(traces.example_id)][:N_QUESTIONS], start=1):
    q = data.test.loc[ex]
    out.append(f"## Question {n} (id {ex})\n")
    out.append(sentences(q) + f"\nCorrect answer: {SPEC.true_label(q)}\n")
    section_case(out, "Original question", ex, traces, expl, data)
    real = traces[(traces.example_id == ex) & (traces.row_type == "real") & (traces.budget == "uncapped")
                  & (traces.sample_idx == 0) & traces["I"].notna()].sort_values("intervention_idx")
    for flipped, title in ((1.0, "Inserted word that changed the trace's answer"),
                           (0.0, "Inserted word that did not change the trace's answer")):
      pick = real[real["I"] == flipped]
      if pick.empty:
        out.append(f"### {title}\n\n(none for this question)\n")
        continue
      r = pick.iloc[0]
      section_case(out, f"{title}: \"{r['inserted_word']}\"", ex, traces, expl, data, iv=r["intervention_idx"])
  path = analyze.OUT / "EXAMPLES.md"
  path.write_text("\n".join(out))
  print("wrote", path.relative_to(analyze.ROOT))


if __name__ == "__main__":
  main()
