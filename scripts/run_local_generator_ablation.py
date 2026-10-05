"""Compare local generators on the full EnterpriseRAG-Bench question set through Ollama.

Every generator answers from the same context: the API's CRAG retrieval
(hybrid + cross-encoder rerank) and the API's prompt (core.generator.build_prompt).
An independent judge model, which is not one of the candidates, grades each
answer in a single call: overall correctness against the gold answer, and which
gold facts the answer states (completeness).

The run is resumable. Questions are shuffled once (seeded) and processed in
chunks; each chunk is retrieved, answered by every model, then judged, and every
result is appended to JSONL files in --run-dir. Re-running with the same
--run-dir skips finished work, so partial results are always a random sample
answered by all models. Use --summarize to aggregate whatever has finished.

Judge reliability is checked by also grading the gold answers themselves
(--judge-sanity questions); a reliable judge marks them correct and complete.
"""

import argparse
import json
import random
import re
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import chromadb

import config
from benchmark import parse_ids
from core.ablation_report import compute_summary, read_jsonl, slug
from core.generator import build_prompt
from core.results_store import save_result
from core.service import RAGService

# model id -> (Ollama tag, context window). Contexts are matched across models,
# so --max-context-chars must fit the smallest window (phi-2: 2048 tokens).
CANDIDATES = {
    "Qwen/Qwen2.5-7B-Instruct": ("qwen2.5:7b-instruct-q4_K_M", 4096),
    "microsoft/Phi-3-mini-4k-instruct": ("phi3:mini-4k", 4096),
    "microsoft/phi-2": ("phi:2.7b", 2048),
}
JUDGE_MODEL = "llama3.1:8b-instruct-q4_K_M"


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")


# A multi-day run must survive Ollama dropping a connection (it did once, 125 questions in,
# and the whole run stopped); a few retries with growing waits cost seconds instead.
OLLAMA_RETRY_WAITS = (15, 30, 60, 120)


def post_json(url: str, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode()
    for attempt, wait in enumerate((*OLLAMA_RETRY_WAITS, None), 1):
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as exc:
            if wait is None:
                raise
            print(f"ollama call failed ({type(exc).__name__}: {exc}); retry {attempt} in {wait}s", flush=True)
            time.sleep(wait)


def ollama_chat(host: str, model: str, prompt: str, *, num_ctx: int, max_tokens: int,
                timeout: int, threads: int | None, json_mode: bool = False, seed: int = 0,
                keep_alive: int | str = 0) -> dict:
    options = {"temperature": 0, "seed": seed, "num_predict": max_tokens, "num_ctx": num_ctx}
    if threads:
        options["num_thread"] = threads
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # Candidates unload after every call: Ollama's server keeps each prompt's KV cache
        # in RAM (up to 8 GB), which OOM-killed phi3 (no GQA, ~600 MB per prompt) on a
        # 14 GB laptop. The judge stays loaded within a question and is unloaded after it.
        "keep_alive": keep_alive,
        "options": options,
    }
    if json_mode:
        body["format"] = "json"
    return post_json(f"{host.rstrip('/')}/api/chat", body, timeout)


def unload(host: str, model: str) -> None:
    post_json(f"{host.rstrip('/')}/api/generate", {"model": model, "keep_alive": 0}, timeout=120)


def load_all_questions(client) -> list[dict]:
    rows = client.get_collection(config.CHROMA_QUESTIONS_COLLECTION).get(
        include=["documents", "metadatas"]
    )
    questions = []
    for question_id, text, metadata in zip(rows["ids"], rows["documents"], rows["metadatas"]):
        metadata = metadata or {}
        questions.append({
            "question_id": question_id,
            "question": text,
            "category": metadata.get("question_type", ""),
            "gold_answer": metadata.get("gold_answer", ""),
            "facts": [fact.strip() for fact in metadata.get("answer_facts", "").split("|") if fact.strip()],
            "gold_doc_ids": parse_ids(metadata.get("expected_doc_ids")),
        })
    return questions


FACTS_PER_JUDGE_CALL = 6


def judge_prompt(question: dict, answer: str, facts: list[str], with_correctness: bool) -> str:
    # Asking for the numbers of stated facts (rather than one boolean per fact) avoids the
    # 8B judge's failure modes: counting past the last fact and marking nearly all true.
    listed = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
    correctness = (
        "- correct: true if the candidate is broadly consistent with the reference answer. "
        "Be lenient about wording and extra detail, strict about conflicting facts, names and quantities. "
        "If the reference says the question cannot be answered from the documents, the candidate is "
        "correct only if it says so.\n"
    ) if with_correctness else ""
    output = '{"correct": true or false, "present": [fact numbers]}' if with_correctness \
        else '{"present": [fact numbers]}'
    return (
        "You grade a candidate answer to a question.\n"
        f"{correctness}"
        f"- present: the numbers (1-{len(facts)}) of the facts below that the candidate answer itself "
        "states. Leave a fact out if the candidate does not state it. Use [] if it states none.\n"
        f"Return JSON only: {output}\n\n"
        f"Question: {question['question']}\n\n"
        # Fact checks need only the facts; repeating a long reference answer in every
        # batch made each call on many-fact questions take minutes on CPU.
        + (f"Reference answer: {question['gold_answer']}\n\n" if with_correctness else "")
        + f"Facts:\n{listed}\n\nCandidate answer: {answer}"
    )


def context_of(prompt: str) -> str:
    """The retrieved context alone, so the question text cannot count as evidence."""
    start = prompt.find("Context:\n")
    end = prompt.rfind("\n\nQuestion:")
    return prompt[start + len("Context:\n"):end] if start != -1 and end > start else prompt


# Fact presence in the model's context is scored by the cross-encoder, not the LLM judge.
# On 25 questions x 142 facts with negative controls (another question's context), the
# 8B judge marked 79% of facts present in the right context and 72% in an unrelated one
# (no separation), while the cross-encoder separated them with AUC 0.884; at 0.1 it
# credits 3% of facts in unrelated contexts. It is conservative: reworded facts may be missed.
CONTEXT_FACT_THRESHOLD = 0.1
DOC_SPLIT = re.compile(r"(?:^|\n\n)Doc \d+: ")


def grade_context(scorer, question: dict, prompt: str) -> dict:
    """Which gold facts reached the model's context: separates retrieval/context loss from generation errors."""
    blocks = [block.strip() for block in DOC_SPLIT.split(context_of(prompt)) if block.strip()]
    scores = [max(scorer([(fact, block) for block in blocks])) if blocks else 0.0 for fact in question["facts"]]
    return {
        "facts_in_context": [score >= CONTEXT_FACT_THRESHOLD for score in scores],
        "fact_scores": [round(float(score), 4) for score in scores],
        "method": "cross-encoder",
        "threshold": CONTEXT_FACT_THRESHOLD,
    }


def strip_citations(answer: str) -> str:
    answer = re.sub(r"\[[^\]]+\]", " ", answer)
    answer = re.sub(r"\bDoc(?:ument)?\s*#?\s*\d+\b", " ", answer, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", answer).strip()


def _judge_call(args, prompt: str, n_facts: int, attempts: int = 2) -> tuple[dict, bool, str]:
    """One judge call, retried with another seed when the output is not usable JSON."""
    text = ""
    for attempt in range(attempts):
        response = ollama_chat(
            args.host, args.judge_model, prompt,
            num_ctx=4096, max_tokens=30 + 4 * n_facts,
            timeout=args.timeout, threads=args.threads, json_mode=True, seed=attempt,
            keep_alive="5m",
        )
        text = response["message"]["content"]
        try:
            verdict = json.loads(re.search(r"\{.*\}", text, flags=re.DOTALL).group(0))
            present = verdict.get("present", [])
            if not isinstance(present, list):
                raise ValueError("present is not a list")
            numbers = {int(item) for item in present}
        except (AttributeError, ValueError, TypeError):
            continue
        correct = verdict.get("correct")
        if isinstance(correct, str):
            correct = correct.strip().lower() == "true"
        return {"correct": correct is True, "present": numbers}, True, text
    return {}, False, text


def judge(args, question: dict, answer: str) -> dict:
    """Correctness plus per-fact presence; long fact lists are graded in small batches,
    since an 8B judge marks every fact present (or emits broken JSON) on 40+ facts at once."""
    answer = strip_citations(answer)
    all_facts = question["facts"]
    batches = [all_facts[i:i + FACTS_PER_JUDGE_CALL] for i in range(0, len(all_facts), FACTS_PER_JUDGE_CALL)] or [[]]
    correct, facts, parse_ok, failed_raw = False, [], True, []
    for index, batch in enumerate(batches):
        verdict, ok, raw = _judge_call(args, judge_prompt(question, answer, batch, index == 0), len(batch))
        parse_ok &= ok
        if not ok:
            failed_raw.append(raw[:300])
        if index == 0:
            correct = bool(verdict.get("correct"))
        # null = the batch could not be graded; completeness is computed over graded facts only.
        facts += [(i in verdict["present"]) if ok else None for i in range(1, len(batch) + 1)]
    return {
        "correct": correct,
        "facts_present": facts,
        "facts_ungraded": sum(fact is None for fact in facts),
        "judge_parse_ok": parse_ok,
        "judge_calls": len(batches),
        "judge_failed_raw": failed_raw,
    }


def run(args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    service = RAGService()
    service.initialize()
    scorer = service.crag.corrector.scorer  # cross-encoder for context-fact checks
    questions = load_all_questions(service.client)
    random.Random(args.seed).shuffle(questions)
    questions = questions[: args.limit]

    manifest = {
        "question_count": len(questions),
        "seed": args.seed,
        "retrieval": f"CRAG: hybrid top-{args.top_k * config.CRAG_RETRIEVAL_MULTIPLIER} reranked by "
                     f"{config.RERANKER_MODEL}, top-{args.top_k} kept; identical for every model",
        "context": f"core.generator.build_prompt, {args.max_context_chars} chars",
        "max_new_tokens": args.max_new_tokens,
        "candidates": {model: tag for model, (tag, _) in CANDIDATES.items()},
        "judge_model": args.judge_model,
        "judge_mode": "correctness + listed present facts, facts graded in batches of "
                      f"{FACTS_PER_JUDGE_CALL}",
        "judge_sanity": args.judge_sanity,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    retrieval_path = run_dir / "retrieval.jsonl"
    sanity_path = run_dir / "judge_sanity.jsonl"
    context_path = run_dir / "context_facts.jsonl"
    context_controls_path = run_dir / "context_controls.jsonl"
    answer_paths = {model: run_dir / f"{slug(model)}.jsonl" for model in CANDIDATES}
    judgment_paths = {model: run_dir / f"{slug(model)}_judged.jsonl" for model in CANDIDATES}
    sanity_ids = {q["question_id"] for q in questions[: args.judge_sanity]}
    started = time.time()

    for start in range(0, len(questions), args.chunk_size):
        chunk = questions[start:start + args.chunk_size]
        retrieved = read_jsonl(retrieval_path)
        for question in chunk:
            if question["question_id"] in retrieved:
                continue
            documents = service.crag.retrieve_corrected(question["question"], args.top_k)
            gold = set(question["gold_doc_ids"])
            ids = [doc["doc_id"] for doc in documents]
            row = {
                "question_id": question["question_id"],
                "document_ids": ids,
                "recall_at_k": len(gold & set(ids)) / len(gold) if gold else None,
                "prompt": build_prompt(question["question"], documents, args.max_context_chars),
            }
            append_jsonl(retrieval_path, row)
            retrieved[question["question_id"]] = row

        for model, (tag, num_ctx) in CANDIDATES.items():
            done = read_jsonl(answer_paths[model])
            todo = [q for q in chunk if q["question_id"] not in done]
            for question in todo:
                t0 = time.time()
                response = ollama_chat(
                    args.host, tag, retrieved[question["question_id"]]["prompt"],
                    num_ctx=num_ctx, max_tokens=args.max_new_tokens,
                    timeout=args.timeout, threads=args.threads,
                )
                prompt_tokens = response.get("prompt_eval_count") or 0
                append_jsonl(answer_paths[model], {
                    "question_id": question["question_id"],
                    "answer": response["message"]["content"].strip(),
                    "seconds": round(time.time() - t0, 2),
                    "prompt_tokens": prompt_tokens,
                    "context_truncated": prompt_tokens + args.max_new_tokens > num_ctx,
                    "hit_token_limit": response.get("eval_count", 0) >= args.max_new_tokens,
                })
            if todo:
                unload(args.host, tag)

        for model in CANDIDATES:
            answers = read_jsonl(answer_paths[model])
            judged = read_jsonl(judgment_paths[model])
            for question in chunk:
                if question["question_id"] in judged:
                    continue
                t0 = time.time()
                verdict = judge(args, question, answers[question["question_id"]]["answer"])
                append_jsonl(judgment_paths[model], {
                    "question_id": question["question_id"], **verdict, "seconds": round(time.time() - t0, 2),
                })
                unload(args.host, args.judge_model)  # bound the judge's prompt cache

        context_done = read_jsonl(context_path)
        controls_done = read_jsonl(context_controls_path)
        for index, question in enumerate(chunk):
            qid = question["question_id"]
            if qid not in context_done:
                t0 = time.time()
                graded = grade_context(scorer, question, retrieved[qid]["prompt"])
                append_jsonl(context_path, {"question_id": qid, **graded, "seconds": round(time.time() - t0, 2)})
            if qid in sanity_ids and qid not in controls_done and len(chunk) > 1:
                # Negative control: another question's context should state ~none of these facts;
                # what the judge credits there is its false-positive rate for context facts.
                other = chunk[(index + 1) % len(chunk)]["question_id"]
                negative = grade_context(scorer, question, retrieved[other]["prompt"])
                append_jsonl(context_controls_path, {"question_id": qid, "context_of": other, **negative})

        sanity_done = read_jsonl(sanity_path)
        for question in chunk:
            if question["question_id"] in sanity_ids and question["question_id"] not in sanity_done:
                # Positive control: the gold answer should be judged correct with all facts present.
                # Negative control: another question's gold answer should be judged incorrect with
                # ~no facts present; what it does get measures the judge's false-positive rate.
                decoy = questions[(questions.index(question) + 1) % len(questions)]["gold_answer"]
                positive = judge(args, question, question["gold_answer"])
                unload(args.host, args.judge_model)
                negative = judge(args, question, decoy)
                unload(args.host, args.judge_model)
                append_jsonl(sanity_path, {"question_id": question["question_id"], **positive,
                                           "negative_control": negative})
        unload(args.host, args.judge_model)

        finished = min(start + args.chunk_size, len(questions))
        print(f"{finished}/{len(questions)} questions done, {(time.time() - started) / 60:.0f} min elapsed", flush=True)

    summarize(run_dir, questions, save=True)


def summarize(run_dir: Path, questions: list[dict], save: bool = False) -> dict:
    summary = compute_summary(run_dir, {q["question_id"]: q["category"] for q in questions})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if save:
        print(f"saved to {save_ablation(run_dir, questions, summary)}")
    return summary


def save_ablation(run_dir: Path, questions: list[dict], summary: dict) -> Path:
    """Register the run in results/: a wide per-question table plus the raw JSONL logs."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    models = list(manifest["candidates"])
    retrieval = read_jsonl(run_dir / "retrieval.jsonl")
    answers = {m: read_jsonl(run_dir / f"{slug(m)}.jsonl") for m in models}
    judged = {m: read_jsonl(run_dir / f"{slug(m)}_judged.jsonl") for m in models}
    rows = []
    for question in questions:
        qid = question["question_id"]
        if qid not in retrieval:
            continue
        row = {"question_id": qid, "category": question["category"], "question": question["question"],
               "gold_answer": question["gold_answer"], "recall_at_k": retrieval[qid]["recall_at_k"]}
        for model in models:
            key = slug(model)
            verdict = judged[model].get(qid)
            row[f"{key}_answer"] = answers[model].get(qid, {}).get("answer")
            row[f"{key}_seconds"] = answers[model].get(qid, {}).get("seconds")
            row[f"{key}_correct"] = None if verdict is None else verdict["correct"]
            row[f"{key}_facts_present"] = None if verdict is None else sum(1 for f in verdict["facts_present"] if f)
            row[f"{key}_facts_total"] = None if verdict is None else len(verdict["facts_present"])
        rows.append(row)
    logs = {p.name: p for p in sorted(run_dir.glob("*.json*")) if p.name != "summary.json"}
    return save_result(
        "generator_ablation",
        f"Generators on {summary['questions_judged_by_all_models']}/{manifest['question_count']} questions "
        f"({run_dir.name})",
        {"run_dir": str(run_dir), "manifest": manifest, "summary": summary},
        headline={m.split("/")[-1]: round(v["leaderboard_score"], 3) for m, v in summary["models"].items()}
        | {"n": summary["questions_judged_by_all_models"]},
        tables={"per_question": rows},
        files=logs,
        label=run_dir.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="data/ablation_runs/full_benchmark")
    parser.add_argument("--limit", type=int, help="first N shuffled questions (default: all)")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=config.TOP_K_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default=config.OLLAMA_HOST)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-context-chars", type=int, default=4500,
                        help="matched context; must fit phi-2's 2048-token window")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="128 cut off ~1 in 3 pilot answers mid-sentence")
    parser.add_argument("--threads", type=int, default=8,
                        help="Ollama CPU threads, leaving some for the rest of the laptop")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-sanity", type=int, default=25)
    parser.add_argument("--summarize", action="store_true", help="only aggregate existing results")
    parser.add_argument("--save", action="store_true", help="with --summarize: also register a snapshot in results/")
    args = parser.parse_args()
    if args.judge_model in {tag for tag, _ in CANDIDATES.values()}:
        parser.error("the judge must not be one of the candidate models")
    if args.summarize:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        summarize(Path(args.run_dir), load_all_questions(client), save=args.save)
        return
    run(args)


if __name__ == "__main__":
    main()
