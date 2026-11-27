"""Score four-choice ARC JSONL on a running N1X server. Writes one JSON record per item."""

import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def score(base, rows):
    def post(path, **body):
        req = Request(
            base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urlopen(req, timeout=1200) as response:
                return json.load(response)
        except HTTPError as err:
            raise RuntimeError(
                f"{path} returned {err.code}: {err.read().decode()}"
            ) from err

    def complete(tokens, count):
        return post(
            "/v1/completions",
            prompt=tokens,
            max_tokens=count,
            cache_prompt=False,
            temperature=0,
            ignore_eos=True,
            seed=1,
            n_probs=5,
        )

    # Check the same token frontier by incremental decode and cold full prefill.
    # Tokenized by ds4: "The days of the week are Monday, Tuesday,"
    prefix = [671, 3137, 294, 270, 3469, 477, 9219, 14, 10056, 14]
    run = complete(prefix, 5)
    generated = [p["id"] for p in run["choices"][0]["logprobs"]["content"]]
    assert len(generated) == 5, run
    replay = complete(prefix + generated[:-1], 1)
    frontier = replay["choices"][0]["logprobs"]["content"][0]
    assert frontier["id"] == generated[-1] and replay["timings"]["cache_n"] == 0, replay
    print(
        json.dumps(
            {"check": "cold_replay", "tokens": generated, "timings": run["timings"]}
        ),
        flush=True,
    )

    letters = "ABCD"
    for row in rows:
        choices = row["choices"]
        if len(choices["text"]) != 4 or len(choices["label"]) != 4:
            raise ValueError(f"expected four choices for {row['id']}")
        answer = letters[choices["label"].index(row["answerKey"])]
        options = "\n".join(
            f"{c}. {opt}" for c, opt in zip(letters, choices["text"], strict=True)
        )
        question = f"{row['question']}\n{options}\nReply with the letter of the correct answer."
        # Equal bias lifts all four answer tokens into the reported distribution.
        # Renormalizing cancels that bias and measures P(answer | one of A-D).
        result = post(
            "/v1/chat/completions",
            messages=[{"role": "user", "content": question}],
            chat_template_kwargs={"enable_thinking": False},
            max_tokens=1,
            seed=1,
            cache_prompt=False,
            temperature=1,
            samplers=["temperature"],
            logprobs=True,
            top_logprobs=4,
            post_sampling_probs=True,
            logit_bias=[[c, 100] for c in letters],
        )
        top = result["choices"][0]["logprobs"]["content"][0]["top_probs"]
        probs = {p["token"]: p["prob"] for p in top}
        assert set(probs) == set(letters) and result["timings"]["cache_n"] == 0, result
        nll = -math.log(probs[answer] / sum(probs.values()))
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "expected": answer,
                    "predicted": max(probs, key=probs.get),
                    "nll": nll,
                    "probabilities": probs,
                    "timings": result["timings"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    with args.questions.open(encoding="utf-8") as questions:
        score(
            args.url.rstrip("/"),
            (json.loads(line) for line in questions if line.strip()),
        )
