"""Exercise the resident N1X endpoint, including streamed tool calls and KV reuse."""
import argparse
import json
import urllib.request

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--switch-chat", action="store_true",
                        help="check RAM prompt restoration across an unrelated chat")
    checks.add_argument("--expect-snapshot-restores", type=int, metavar="COUNT",
                        help="run the 14-request eviction trace against a freshly started server")
    args = parser.parse_args()
    with urllib.request.urlopen(BASE + "/models", timeout=5) as response:
        model = json.load(response)["data"][0]["id"]
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
