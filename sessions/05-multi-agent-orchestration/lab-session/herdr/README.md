# Optional: three short herdr demos

[herdr](https://herdr.dev) runs several terminals, and the agents inside them, in one window. It calls no model itself: it hosts terminals. Nothing here is needed for the lab.

## Install and start

```bash
brew install herdr                               # macOS
curl -fsSL https://herdr.dev/install.sh | sh     # Linux or macOS
```

Windows, in PowerShell: `powershell -ExecutionPolicy Bypass -c "irm https://herdr.dev/install.ps1 | iex"`

Then, from the `lab-session` folder: `herdr`

| Action | Keys (press `ctrl+b`, release, then the key) |
|---|---|
| Split side by side / stacked | `prefix v` / `prefix -` |
| Move between panes | `prefix h` `j` `k` `l` |
| Zoom one pane | `prefix z` |
| Detach (everything keeps running) | `prefix q`, then `herdr` to come back |
| All the keys | `prefix ?` |

The mouse works too: click a pane, drag a border, right-click to split.

Detaching keeps the processes running. `herdr server stop` or a server restart ends them: herdr restores the layout, not the processes.

## Demo 1: three designs side by side

Split the window into three panes (`prefix v` twice), then run one command per pane:

```bash
uv run python herdr/ask.py single
uv run python herdr/ask.py supervisor
uv run python herdr/ask.py handoffs
```

Same question, three designs, running at the same time. Each line is one step, coloured by agent. Look at the end of each pane: how many model calls, how many tokens?

## Demo 2: leave and come back

In one pane, start the four questions:

```bash
uv run python herdr/ask.py supervisor --all
```

Press `prefix q`: the window closes, but the run goes on. Type `herdr`: you are back, and the run has moved on without you.

## Demo 3: coding agents and their status

This is what herdr is built for. Open two panes and start a coding agent in each, if you have one installed (Claude Code or Codex, for example):

```bash
claude "Explain what lab.py does, in five lines"
claude "Write a test for lab.calculator in test_calculator.py"
```

herdr recognises these agents and marks each pane **working**, **blocked** (the agent waits for your approval, for example to write a file) or **idle**. The second agent usually asks before writing a file: its pane turns blocked until you answer. You are now the supervisor of two agents.

Plain scripts, like `herdr/ask.py`, show up as ordinary terminals: herdr only reads the status of the coding agents it knows.
