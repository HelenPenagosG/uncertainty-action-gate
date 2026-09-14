from __future__ import annotations

import argparse
import json

try:
    from .sandbox import Sandbox, TASK
except ImportError:  # Direct execution from this directory.
    from sandbox import Sandbox, TASK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["reset", "task", "exec", "score", "stop"])
    parser.add_argument("action", nargs="?")
    parser.add_argument("--condition", default="B", choices=["A", "B", "C", "C-UNC", "C-COMP"])
    parser.add_argument("--env-variant", choices=["v0", "v1", "v2", "v3"])
    args = parser.parse_args()

    box = Sandbox(condition=args.condition, env_variant=args.env_variant)
    if args.command == "reset":
        result = box.reset()
    elif args.command == "task":
        print(TASK)
        return
    elif args.command == "exec":
        if not args.action:
            parser.error("exec requires an action")
        result = box.execute(args.action)
    elif args.command == "score":
        result = box.score()
    else:
        box.stop()
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
