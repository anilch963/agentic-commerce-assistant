"""Chat with the assistant in your terminal:  python -m app.cli  (add --trace to see agents and tools)."""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid

from app.graph import chat


async def main(show_trace: bool) -> None:
    session = uuid.uuid4().hex[:8]
    print("iPhone store assistant. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue
        result = await chat(message, session)
        print(f"\nbot> {result['answer']}\n")
        if show_trace:
            for step in result["trace"]:
                tools = ", ".join(step["tools"]) or "-"
                print(f"     [{step['agent']:<14}] {step['ms']:>6} ms  tools: {tools}")
            print()


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    asyncio.run(main("--trace" in sys.argv))
