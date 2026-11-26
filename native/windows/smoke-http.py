"""Exercise the resident N1X endpoint, including streamed tool calls and KV reuse."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from threading import Barrier
from time import monotonic
import urllib.request
import uuid

BASE = "http://127.0.0.1:8081/v1"


def complete(body):
    request = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=1200)


def snapshot_prompts():
    # Distinct token histories with one shared BOS and lengths 32 through 72.
    return {key: [0] + list(range(1000 + i * 100, 1031 + i * 100 + 8 * i))
            for i, key in enumerate("ABCDEF")}


def check_snapshot_restores(model, expected):
    prompts = snapshot_prompts()
    restored = cached = 0
    for key in "ABACADAEBFAECA":
        payload = dict(model=model, prompt=prompts[key], temperature=0, max_tokens=1, seed=7)
        req = urllib.request.Request(BASE + "/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as response:
            reply = json.load(response)
        reused = reply["timings"]["cache_n"]
        restored += reused > 1
        cached += reused
    assert restored == expected, (restored, expected, cached)
    print(f"Restored {restored} snapshots across 14 requests and reused {cached} tokens.")


@dataclass
class StreamResult:
    text: str
    prompt: int
    cached: int
    prefill_ms: float
    start: float
    first: float
    last: float

    def report(self, label):
        print(f"{label}: {self.prompt} prompt, {self.cached} cached, "
              f"{self.first - self.start:.2f}s to first content, {self.prefill_ms:.0f} ms prefill", flush=True)


def stream_chat(model, messages, max_tokens):
    parts = []
    first = last = None
    usage = timings = None
    start = monotonic()
    with complete(dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                       stream=True, stream_options={"include_usage": True})) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                text = choice["delta"].get("content")
                if text:
                    last = monotonic()
                    if first is None:
                        first = last
                    parts.append(text)
        else:
            raise AssertionError("Stream ended before [DONE]")
    if not parts or not usage or not timings:
        raise AssertionError(f"Missing content, usage or timings: {usage}, {timings}")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return StreamResult("".join(parts), usage["prompt_tokens"], cached, timings["prompt_ms"], start, first, last)


def check_concurrent_chats(model, parent_lines):
    # Distinguish this run from any history left on a warm server.
    run = uuid.uuid4().hex
    markers = ["CEDAR", "COBALT", "MAPLE", "AMBER"]
    histories = []
    initial_tokens = []
    notes = "".join(f"Record {j}: verified the build and saved its log.\n" for j in range(parent_lines))
    for i, marker in enumerate(markers):
        content = (f"{marker} {run}\n" + (notes if i == 0 else "") +
                   f"Your private word is {marker}. Reply with just your private word.")
        history = [{"role": "user", "content": content}]
        result = stream_chat(model, history, 12)
        result.report(f"{marker} initial")
        if marker not in result.text:
            raise AssertionError(f"{marker} initial answer: {result.text!r}")
        initial_tokens.append(result.prompt)
        history.extend([{"role": "assistant", "content": result.text},
                        {"role": "user", "content": "Repeat your private word twenty times, separated by spaces."}])
        histories.append(history)

    barrier = Barrier(len(markers))

    def follow_up(history):
        barrier.wait(timeout=15)
        return stream_chat(model, history, 48)

    with ThreadPoolExecutor(max_workers=len(markers)) as pool:
        results = list(pool.map(follow_up, histories))
    for marker, original, result in zip(markers, initial_tokens, results):
        result.report(f"{marker} return")
        if marker not in result.text or any(m in result.text for m in markers if m != marker):
            raise AssertionError(f"{marker} returned a missing or mixed private word: {result.text!r}")
        # Allow the assistant generation marker at the end of the old prompt to change.
        if result.cached < original - 8:
            raise AssertionError(f"{marker} reused only {result.cached} of {original} original tokens")
    if max(r.first for r in results) >= min(r.last for r in results):
        raise AssertionError("The four response streams did not deliver content concurrently")
    print("All four streams overlapped and retained their own conversation prefixes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--concurrent", action="store_true",
                        help="check four overlapping agent streams and reuse after unrelated chats")
    parser.add_argument("--parent-lines", type=int, default=64,
                        help="number of synthetic repository notes in the concurrent test's parent prompt")
    checks.add_argument("--switch-chat", action="store_true",
                        help="check RAM prompt restoration across an unrelated chat")
    checks.add_argument("--expect-snapshot-restores", type=int, metavar="COUNT",
                        help="run the 14-request eviction trace against a freshly started server")
    args = parser.parse_args()
    if args.parent_lines < 1:
        parser.error("--parent-lines must be positive")
    with urllib.request.urlopen(BASE + "/models", timeout=5) as response:
        model = json.load(response)["data"][0]["id"]
    if args.concurrent:
        check_concurrent_chats(model, args.parent_lines)
        return
    if args.expect_snapshot_restores is not None:
        check_snapshot_restores(model, args.expect_snapshot_restores)
        return
    messages = [{"role": "user", "content": "Call lookup with key abc. Then reply with just the value it returns."}]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up a key",
              "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
    base = dict(model=model, messages=messages, tools=tools, temperature=0, max_tokens=128)
    with complete(dict(base, stream=True, stream_options={"include_usage": True})) as response:
        assert response.headers.get_content_type() == "text/event-stream"
        call_id = name = arguments = finish = ""
        done = False
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                for call in choice["delta"].get("tool_calls", []):
                    call_id = call.get("id") or call_id
                    name += call.get("function", {}).get("name", "")
                    arguments += call.get("function", {}).get("arguments", "")
    assert done and finish == "tool_calls" and call_id and name == "lookup", (done, finish, call_id, name)
    assert json.loads(arguments) == {"key": "abc"}, arguments
    print(f"Streamed {name}({arguments})", flush=True)
    if args.switch_chat:
        other = [{"role": "user", "content": "Return the digit 7."}]
        with complete(dict(model=model, messages=other, temperature=0, max_tokens=8)) as response:
            assert json.load(response)["usage"]["completion_tokens"] > 0
    messages += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "72519"}]
    with complete(base) as response:
        reply = json.load(response)
    answer = reply["choices"][0]["message"]["content"].strip()
    cached = reply["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    assert "72519" in answer and cached > 0, (answer, cached, reply.get("timings"))
    if args.switch_chat:
        # Restoring the tool-bearing prompt should recover much more than the common BOS.
        assert cached >= 128, (cached, reply.get("timings"))
    print(f"Tool result returned {answer!r}; reused {cached} prompt tokens.")


if __name__ == "__main__":
    main()
