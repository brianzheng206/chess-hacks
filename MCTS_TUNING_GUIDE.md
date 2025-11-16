# MCTS Parameter Tuning Guide

This guide explains how to tune MCTS parameters to reduce blunders and improve play quality.

## Key Parameters

### 1. `c_puct` (PUCT Exploration Constant)
**Location**: `src/main.py` line 99, and phase-specific adjustments around lines 400, 404, 408

**What it does**: Controls the balance between:
- **Exploitation** (trusting moves with high Q-values)
- **Exploration** (trying moves with high policy probabilities)

**Current values**:
- Default: `0.8`
- Opening: `1.4`
- Midgame: `0.8`
- Endgame: `0.7`

**How to tune**:
- **Lower values (0.5-0.8)**: Trust policy more, less exploration
  - Use when: Model is well-trained, you see MCTS choosing bad moves over good policy moves
  - Effect: More conservative, follows model's intuition
  - Risk: May miss tactical shots that require deeper search

- **Higher values (1.0-2.0)**: More exploration, less trust in policy
  - Use when: Model is weak, you want more search depth
  - Effect: More aggressive exploration, may find hidden tactics
  - Risk: May explore bad lines and blunder

**Example adjustments**:
```python
# If MCTS keeps choosing moves with <5% policy over moves with >30% policy:
c_puct = 0.6  # Trust policy even more

# If you're missing tactical shots:
c_puct = 1.0  # Explore more
```

---

### 2. `sims` (Number of Simulations)
**Location**: `src/main.py` line 98, and phase-specific caps around lines 398, 402, 406

**What it does**: Number of MCTS simulations to run. More simulations = stronger but slower.

**Current values**:
- Default: `200`
- Opening: `120` (capped)
- Midgame: `200` (capped)
- Endgame: `150` (capped)

**How to tune**:
- **More simulations (300-500)**: Better convergence, fewer blunders
  - Use when: You have time, want maximum strength
  - Effect: MCTS has more time to explore and find best moves
  - Tradeoff: Slower moves

- **Fewer simulations (100-150)**: Faster moves, may blunder more
  - Use when: Time-constrained, or in obvious positions
  - Effect: Faster but less reliable
  - Tradeoff: May not converge on best move

**Example adjustments**:
```python
# If you see blunders in complex positions:
sims = 300  # More simulations for better convergence

# If moves are too slow:
sims = 150  # Faster but less reliable
```

---

### 3. Policy Trust Override Thresholds
**Location**: `src/main.py` lines 625-632

**What it does**: Safety net that overrides MCTS if it chooses a move the model strongly dislikes.

**Current thresholds**:
```python
top_policy_prob > 0.15 and           # Top move has >15% probability
mcts_policy_prob < 0.05 and          # MCTS choice has <5% probability
top_policy_prob > mcts_policy_prob * 3.0  # Top move is 3x more likely
```

**How to tune**:
- **More aggressive (lower thresholds, higher multiplier)**:
  ```python
  # Trust policy more aggressively
  if top_policy_prob > 0.10 and mcts_policy_prob < 0.08 and top_policy_prob > mcts_policy_prob * 2.5:
  ```
  - Use when: Model is very reliable, MCTS blunders often
  - Effect: More overrides, more conservative play

- **Less aggressive (higher thresholds, lower multiplier)**:
  ```python
  # Only override in extreme cases
  if top_policy_prob > 0.25 and mcts_policy_prob < 0.03 and top_policy_prob > mcts_policy_prob * 5.0:
  ```
  - Use when: Model is weak, want to trust MCTS more
  - Effect: Fewer overrides, more exploration

---

### 4. Phase-Specific Adjustments
**Location**: `src/main.py` lines 397-408

**What it does**: Different parameters for opening/midgame/endgame based on game phase.

**Current settings**:
```python
if game_phase > 0.7:  # Opening
    sims = min(sims, 120)
    c_puct = 1.4  # More exploration in opening
elif game_phase < 0.3:  # Endgame
    sims = min(sims, 150)
    c_puct = 0.7  # Trust policy more in endgame
else:  # Midgame
    sims = min(sims, 200)
    c_puct = 0.8
```

**How to tune**:
- **Opening**: Usually book moves, less search needed
  - Lower `sims`, higher `c_puct` (exploration is okay)
  
- **Endgame**: Model is usually accurate, trust it more
  - Lower `c_puct` (0.6-0.7), moderate `sims`
  
- **Midgame**: Most complex, need balance
  - Moderate `c_puct` (0.8-1.0), higher `sims`

---

## Common Blunder Scenarios & Fixes

### Scenario 1: MCTS chooses low-probability moves over high-probability moves
**Symptoms**: 
- Policy says move A has 40% probability, move B has 2%
- MCTS chooses move B
- Move B is a blunder

**Fixes**:
1. **Lower `c_puct`** (e.g., 0.6-0.7)
2. **Make policy trust override more aggressive**:
   ```python
   if top_policy_prob > 0.10 and mcts_policy_prob < 0.08 and top_policy_prob > mcts_policy_prob * 2.5:
   ```
3. **Increase simulations** (e.g., 300-400) so MCTS has more time to converge

---

### Scenario 2: Missing tactical shots
**Symptoms**:
- Model doesn't see a tactic (low policy probability)
- MCTS also doesn't find it
- You lose a winning position

**Fixes**:
1. **Increase `c_puct`** (e.g., 1.0-1.2) to explore more
2. **Increase simulations** (e.g., 400-600) for deeper search
3. **Less aggressive policy trust override** (higher thresholds)

---

### Scenario 3: Blunders in time pressure
**Symptoms**:
- Moves are too slow
- When time is low, makes bad moves

**Fixes**:
1. **Reduce default `sims`** (e.g., 150)
2. **Add time-based simulation scaling**:
   ```python
   if ctx.timeLeft < 5000:  # Less than 5 seconds
       sims = min(sims, 100)
   ```
3. **Lower `c_puct`** in time pressure to trust policy more

---

### Scenario 4: Blunders in specific game phases
**Symptoms**:
- Blunders only in opening/endgame/midgame

**Fixes**:
1. **Adjust phase-specific `c_puct`**:
   ```python
   if game_phase > 0.7:  # Opening
       c_puct = 1.0  # Less exploration
   elif game_phase < 0.3:  # Endgame
       c_puct = 0.6  # Trust policy even more
   ```
2. **Adjust phase-specific `sims`**:
   ```python
   if game_phase < 0.3:  # Endgame
       sims = min(sims, 200)  # More simulations in endgame
   ```

## Quick Tuning Reference

| Problem | Parameter | Direction | Example Value |
|---------|-----------|-----------|---------------|
| MCTS chooses bad moves over good policy | `c_puct` | Lower | 0.6-0.7 |
| Missing tactical shots | `c_puct` | Higher | 1.0-1.2 |
| Too many blunders | `sims` | Higher | 300-400 |
| Moves too slow | `sims` | Lower | 100-150 |
| Policy trust override too aggressive | Override thresholds | Less strict | `> 0.25, < 0.03, > 5.0x` |
| Policy trust override not catching blunders | Override thresholds | More strict | `> 0.10, < 0.08, > 2.5x` |

---

## Testing Your Changes

1. **Play test games** and note when blunders occur
2. **Check the logs** for:
   - Policy probabilities vs MCTS choice
   - Whether policy trust override triggered
   - Search time and simulations
3. **Adjust incrementally** - change one parameter at a time
4. **Keep a log** of what works and what doesn't

---

## Advanced: Dynamic Tuning

You can make parameters adaptive based on position characteristics:

```python
# Trust policy more when model is very confident
if top_policy_prob > 0.5:
    c_puct = 0.6  # Very confident, trust it
elif top_policy_prob > 0.3:
    c_puct = 0.8  # Moderately confident
else:
    c_puct = 1.0  # Uncertain, explore more

# More simulations in complex positions
if len(legal_moves) > 30:  # Many legal moves = complex
    sims = min(sims * 1.5, 400)
```

---

## Recommended Starting Points

**Conservative (fewer blunders, may miss tactics)**:
- `c_puct = 0.7`
- `sims = 250`
- Policy trust override: `> 0.12, < 0.06, > 3.0x`

**Balanced (current settings)**:
- `c_puct = 0.8`
- `sims = 200`
- Policy trust override: `> 0.15, < 0.05, > 3.0x`

**Aggressive (more exploration, may find tactics)**:
- `c_puct = 1.0`
- `sims = 300`
- Policy trust override: `> 0.20, < 0.03, > 4.0x`

