# ♟ CHESS-KILLER

**CHESS-KILLER** is an experimental chess engine built from scratch in Python, combining
custom chess rules implementation and a neural network–based evaluation function.

The goal is not to re-implement Stockfish, but to explore:

* how a chess engine can be written end-to-end,
* how learning emerges from self-play,
* and how neural evaluation behaves without hard-coded heuristics.

---

## Project Goals

* Implement a fully functional chess engine (rules, legality, end conditions)
* Learn a position evaluation function using a neural network
* Train the engine via self-play and external opponents
* Experiment with anti-repetition mechanisms and temporal context

This project is research-oriented and intentionally iterative.

---

## Architecture Overview

### 1. Chess Board Modelization ♟

* Board represented as an 8×8 matrix
* Pieces encoded as signed numeric values

  * Sign → color (White / Black)
  * Magnitude → piece type & relative value

```
Pawn   = 1
Knight = 2.7
Bishop = 3
Rook   = 5
Queen  = 9
King   = 20
```

* Positive values represent White
* Negative values represent Black

This representation allows the board to be directly flattened and fed into a neural network.

---

### 2. Move Generation and Game Rules

All chess rules are implemented manually (no chess libraries):

* Legal move generation per piece
* Captures
* Check and checkmate detection
* Stalemate
* En passant
* Threefold repetition
* Fifty-move rule
* Full validation of illegal moves (including king safety)

Core logic lives in:

* `canTake`
* `isCheck / isCheck_`
* `isValidMove`
* `getAllValidMove`

---

### 3. Neural Network Evaluation

Each position is evaluated by a PyTorch neural network.

**Input**

* Flattened board (64 values)
* Temporal context: last N moves encoded as additional features

This allows the network to:

* detect repetitions,
* reason about recent move history,
* reduce naive loops.

**Output**

* A single scalar in the range `[-1, 1]`

  * `+1` → winning position
  * `0` → equal position
  * `-1` → losing position

---

### 4. Players

* **HumanPlayer**
  Manual input for testing and debugging.

* **Player (AI)**

  * Evaluates all legal moves
  * Selects the best move based on network evaluation
  * Applies a dynamic repetition penalty

---

### 5. Training Strategy

The engine learns via reinforcement learning:

* Self-play or human play
* Final game outcome used as reward
* Discounted rewards with γ = 0.99
* Backpropagation through all visited positions

```
Win  → +1
Draw →  0
Loss → -1
```

---

## Known Issues and Current Behavior

* During self-play, the engine naturally converges to repeated positions
* This is expected: identical positions yield identical evaluations
* Explored solutions:

  * repetition penalty
  * temporal move encoding (currently implemented)
  * extended move horizon

This behavior is intentional and part of the experimentation.

---

## Planned Improvements

* Stronger repetition discouragement
* Larger temporal window
* Minimax or MCTS hybrid search
* Training against Stockfish
* Improved reward shaping
* GPU training support
* Opening diversity injection

---

## Dependencies

```bash
pip install torch
```

Python 3.9 or higher is recommended.

---

## How to Run

```bash
python main.py
```

You can:

* play against the AI
* watch AI vs AI games
* train the network through repeated games

---

## Disclaimer

This is not a production chess engine.
It is a learning and experimentation project focused on:

* chess mechanics
* neural evaluation
* reinforcement learning dynamics

---

## Author

Built as a research and experimentation project.

---
