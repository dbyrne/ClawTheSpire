# ClawTheSpire

An autonomous AI bot for [Slay the Spire 2](https://store.steampowered.com/app/2868840/Slay_the_Spire_2/) that plays the game end-to-end — from character select through boss fights. It combines an exhaustive combat solver with an LLM-powered strategic advisor, all orchestrated through a real-time terminal UI.

Built with [Claude Code](https://claude.ai/claude-code) + the [STS2-Agent](https://github.com/CharTyr/STS2-Agent) game mod.

## How It Works

```
Slay the Spire 2 (game)
        │  HTTP API (mod)
        ▼
   Game Client ──► Runner (TUI game loop)
        │              │
        ├──► Combat Solver ──► Evaluator
        │    (exhaustive search)   (state scoring)
        │
        └──► Strategic Advisor
             (local LLM via Ollama)
             Card rewards, map, shop, events, rest sites
```

**Combat** is handled by a depth-first solver that evaluates every legal card-play ordering and picks the highest-scoring sequence. The evaluator scores states based on damage dealt, enemy threat priority, block efficiency, power/buff values, and character-specific weights.

**Strategy** (everything outside combat) is handled by an LLM advisor with character-aware prompts, card tier lists, relic synergy guides, and deck archetype detection.

The **runner** ties it all together in a polling loop — detecting the current screen, dispatching to the right decision engine, executing actions, and logging everything.

## Features

- **Two characters**: Ironclad and The Silent, with character-specific strategy, tier lists, and evaluator tuning
- **Rich terminal UI**: Three-panel layout showing run status, solver output, advisor reasoning, and a scrolling action log
- **Batch runner**: Continuous game loop with hot-reloadable config, automatic dashboard updates, and per-character stats
- **Event-sourced logging**: Full JSONL logs capturing every decision, state change, and outcome for replay and analysis
- **Progress dashboard**: Auto-deployed Chart.js dashboard tracking floor progression across generations of improvements
- **MCP server integration**: Both the game interface and solver expose [MCP](https://modelcontextprotocol.io/) tools, so Claude Code (or any MCP client) can play interactively

## Architecture

```
STS2/
├── run.py                      # Entry point for interactive runner
├── sts2_config.json            # Bot configuration
├── sts2-solver/                # Core bot logic (Python)
│   └── src/sts2_solver/
│       ├── runner.py           # Autonomous game loop + TUI
│       ├── batch_runner.py     # Continuous multi-game runner
│       ├── solver.py           # Exhaustive combat search
│       ├── combat_engine.py    # Turn simulation engine
│       ├── evaluator.py        # Combat state scoring
│       ├── advisor.py          # LLM strategic advisor
│       ├── advisor_prompts.py  # Character-aware prompt builder
│       ├── config.py           # Tunable weights, tier lists, strategy params
│       ├── bridge.py           # Game state ↔ solver state conversion
│       ├── game_client.py      # HTTP client to game mod
│       ├── run_logger.py       # Event-sourced JSONL logging
│       ├── mcp_server.py       # MCP tools: solve_combat, advise_strategy
│       └── ...
├── dashboard/                  # Vercel-hosted progress dashboard
│   ├── index.html              # Chart.js visualization
│   └── update_data.py          # Log scanner → data.json
└── STS2-Agent/                 # Game mod (git submodule, not included)
```

## Prerequisites

- **Slay the Spire 2** (Steam)
- **[STS2-Agent mod](https://github.com/CharTyr/STS2-Agent)** installed in the game's mods directory
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **[Ollama](https://ollama.com/)** running locally with a model pulled (e.g. `ollama pull qwen3:8b`)

## Setup

```bash
# Clone with the game mod submodule
git clone --recurse-submodules https://github.com/dbyrne/ClawTheSpire.git
cd ClawTheSpire

# Install Python dependencies
cd sts2-solver
uv sync
cd ..

# Pull the advisor LLM
ollama pull qwen3:8b

# Start the game, enable the STS2AIAgent mod, and verify:
curl http://127.0.0.1:8081/health
```

## Usage

### Single Game (Interactive TUI)

```bash
python run.py                        # Auto-play as Ironclad
python run.py --character Silent     # Play as The Silent
python run.py --step                 # Step mode: press Enter per action
python run.py --dry-run              # Show decisions without executing
```

### Batch Runner (Continuous)

```bash
python -m sts2_solver.batch_runner              # Run games forever
python -m sts2_solver.batch_runner --once       # Single game, then exit
python -m sts2_solver.batch_runner --character Silent
```

The batch runner reads `sts2_config.json` before every game, so you can change config mid-session:

```json
{
  "gen": 6,
  "character": "Ironclad",
  "characters": ["Ironclad", "Silent"],
  "model": "qwen3:14b",
  "poll_interval": 1.0,
  "deploy_dashboard": true
}
```

Set `characters` to rotate between characters each game. The `--character` CLI flag overrides everything.

### MCP Servers (for Claude Code)

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "sts2-ai-agent": {
      "command": "uv",
      "args": ["run", "--directory", "STS2-Agent/mcp_server", "sts2-mcp-server"],
      "env": { "STS2_API_BASE_URL": "http://127.0.0.1:8081" }
    },
    "sts2-solver": {
      "command": "uv",
      "args": ["run", "--directory", "sts2-solver", "sts2-solver-mcp"],
      "env": {
        "STS2_API_BASE_URL": "http://127.0.0.1:8081",
        "STS2_ADVISOR_BASE_URL": "http://localhost:11434/v1",
        "STS2_ADVISOR_MODEL": "qwen3:8b"
      }
    }
  }
}
```

Then Claude Code can play the game interactively using `solve_combat` and `advise_strategy` tools.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STS2_API_BASE_URL` | `http://127.0.0.1:8081` | Game mod HTTP endpoint |
| `STS2_ADVISOR_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `STS2_ADVISOR_MODEL` | `qwen3:8b` | LLM model for strategic decisions |
| `STS2_ADVISOR_MAX_TOKENS` | `256` | Max tokens for advisor responses |

### Tuning

All solver and evaluator weights live in `sts2-solver/src/sts2_solver/config.py`:

- **Evaluator weights**: Kill bonuses, threat multipliers, block scaling, energy penalties
- **Power values**: Per-character scoring for buffs/powers (e.g. Demon Form, Accuracy)
- **Card tier lists**: S/A/B/Avoid tiers per character, used in advisor prompts
- **Strategy params**: Deck size targets, HP thresholds, boss floor locations

## How the Solver Works

The combat solver does a depth-first search over all legal card-play orderings for the current turn:

1. **Generate legal plays** — cards in hand with enough energy, valid targets
2. **Simulate each play** — update player/enemy state (damage, block, powers, draw)
3. **Recurse** — after each play, generate the next set of legal plays
4. **Evaluate leaf states** — score the resulting board position
5. **Return the best sequence** — highest-scoring path through the search tree

Key optimizations:
- Deduplicates equivalent plays (two copies of Strike → only try one)
- Respects time budget (soft cap, default 5s)
- Prunes branches that can't improve on the current best
- Character-aware power valuations (Ironclad values Demon Form; Silent values Accuracy)

## Dashboard

The bot auto-deploys a progress dashboard after each run showing floor progression across generations:

```bash
python dashboard/update_data.py          # Rebuild data.json from logs
python dashboard/update_data.py --deploy # Also deploy to Vercel
```

## License

MIT
