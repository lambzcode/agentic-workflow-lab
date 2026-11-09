"""Run one design on one question and print every step as it happens: made to watch in a herdr pane.

From the lab-session folder:
    uv run python herdr/ask.py single
    uv run python herdr/ask.py supervisor
    uv run python herdr/ask.py handoffs
    uv run python herdr/ask.py supervisor "How many times larger is Germany's area than Belgium's?"
    uv run python herdr/ask.py supervisor --all          # the four questions
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lab      # noqa: E402
import studio   # noqa: E402  the finished supervisor team and handoff team

DESIGNS = {"single": "single", "supervisor": studio.supervisor_team, "handoffs": studio.handoffs}
ANSI = {"supervisor": "35", "researcher": "34", "analyst": "33", "writer": "32", "agent": "37"}
_record = lab._log


def live(agent, kind, detail, tokens=0, prompt=""):
    """Record the step as usual, and print it now."""
    _record(agent, kind, detail, tokens, prompt)
    read = f"   ({tokens:,} tokens read)" if kind == "model" else ""
    print(f"\033[1;{ANSI.get(agent, '37')}m{agent:<11}\033[0m {kind:<8} {detail}{read}", flush=True)


lab._log = live

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    design = args[0] if args else "supervisor"
    if design not in DESIGNS:
        sys.exit(f"Choose a design: {', '.join(DESIGNS)}")
    questions = [q["q"] for q in lab.QUESTIONS] if "--all" in sys.argv else [args[1] if len(args) > 1 else lab.FIRST]
    for question in questions:
        lab.RUN["meter"] = lab.Meter()
        print(f"\n{design.upper()}: {question}\n")
        try:
            answer = lab._answer(DESIGNS[design], question)
        except Exception as e:                               # a failed call ends this question, not the demo
            answer = f"ERROR: {e}"
        m = lab.RUN["meter"]
        print(f"\nAnswer: {answer}\n{m.calls} model calls, {m.prompt + m.completion:,} tokens, "
              f"{time.time() - m.t0:.0f} s")
