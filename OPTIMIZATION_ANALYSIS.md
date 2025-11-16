# Optimization Analysis for chess_policy and src

## Speed Optimizations

### 1. **Reduce Redundant Board Copies** (High Impact)
**Location**: `src/main.py`, `src/chess_policy/uci.py`

**Issue**: Many unnecessary `board.copy()` calls (20+ instances found). Board copying is expensive.

**Optimizations**:
- Use `push()`/`pop()` pattern instead of copying when possible (already done in `mcts.py` run_simulation)
- Cache board state in SearchNode to avoid repeated copies
- In `main.py` lines 814, 837, 878, 884, 899, 913, 941, 947, 984, 1098, 1199: Many of these copies are only used to push a move and check FEN - use push/pop instead
- In `uci.py` similar pattern - replace with push/pop

**Expected Speedup**: 10-20% reduction in move time, especially in positions with many legal moves

### 2. **Cache Legal Moves and Legal Mask** (High Impact)
**Location**: `src/chess_policy/mcts.py` (expand_node, mcts_search), `src/main.py`

**Issue**: 
- `list(state.generate_legal_moves())` called multiple times per position
- `legal_mask_4672(state)` computed multiple times
- `legal_indices(board)` iterates through all legal moves each time

**Optimizations**:
- Cache legal moves in SearchNode (add `legal_moves` attribute)
- Cache legal mask in SearchNode or BatchEvaluator
- Pre-compute move-to-index mapping once per position

**Expected Speedup**: 5-15% reduction in MCTS overhead

### 3. **Optimize Move-to-Index Mapping** (Medium Impact)
**Location**: `src/chess_policy/mcts.py` (expand_node lines 1064-1078)

**Issue**: 
- Loops through all legal moves and calls `move_to_index()` for each
- Rebuilds move_probs list multiple times when handling duplicates
- Inefficient duplicate detection using `next()` generator

**Optimizations**:
```python
# Instead of:
for move in legal_moves:
    idx = move_to_index(state, move)
    existing_prob = next((p for m, p in move_probs if m == move), None)
    if existing_prob is None or prob > existing_prob:
        move_probs = [(m, p) for m, p in move_probs if m != move]
        move_probs.append((move, prob))

# Use:
move_to_idx_map = {}  # move -> list of indices
for move in legal_moves:
    idx = move_to_index(state, move)
    if idx is not None:
        if move not in move_to_idx_map:
            move_to_idx_map[move] = []
        move_to_idx_map[move].append(idx)

# Then build priors taking max over indices
priors = {}
for move, indices in move_to_idx_map.items():
    max_prob = max(probs[i].item() for i in indices if 0 <= i < len(probs))
    priors[move] = max(max_prob, epsilon)
```

**Expected Speedup**: 3-8% reduction in node expansion time

### 4. **Eliminate Redundant Policy Re-evaluation** (High Impact)
**Location**: `src/chess_policy/mcts.py` (mcts_search lines 1590-1609)

**Issue**: Policy logits are recomputed at root even though root was already expanded with the same evaluation.

**Optimizations**:
- Store policy_logits in SearchNode when expanding
- Reuse stored policy_logits instead of re-evaluating
- Only re-evaluate if Dirichlet noise was applied (which modifies priors)

**Expected Speedup**: 5-10% reduction in search time (saves one model forward pass per search)

### 5. **Optimize FEN String Computations** (Medium Impact)
**Location**: Throughout codebase (28+ FEN calls found)

**Issue**: FEN strings computed multiple times for same position, used for caching/comparison.

**Optimizations**:
- Cache FEN in SearchNode (already done in line 255, but not consistently used)
- Use transposition_key() instead of FEN for comparisons when possible (faster)
- Cache FEN in BatchEvaluator results_cache key computation

**Expected Speedup**: 2-5% reduction in overhead

### 6. **Improve Batch Evaluator Efficiency** (High Impact)
**Location**: `src/chess_policy/mcts.py` (BatchEvaluator class)

**Issue**: 
- Currently evaluates one state at a time (batch_size effectively 1 in single-threaded mode)
- `should_eval_now()` returns True for any pending state, preventing true batching
- No accumulation of leaves before evaluation

**Optimizations**:
- Refactor MCTS to collect multiple leaves before batch evaluation
- Accumulate N leaves (e.g., 8-16) before calling evaluate_batch()
- This requires changing run_simulation to not evaluate immediately, but queue for batch

**Expected Speedup**: 2-4x GPU utilization improvement, 30-50% faster inference when batching 8-16 positions

### 7. **Optimize Legal Mask Tensor Creation** (Low-Medium Impact)
**Location**: `src/chess_policy/mcts.py`, `src/chess_policy/infer.py`

**Issue**: 
- `torch.from_numpy(legal_mask).to(device)` creates new tensor each time
- Could cache on device if same position evaluated multiple times

**Optimizations**:
- Cache legal_mask_tensor in SearchNode or BatchEvaluator
- Pre-allocate tensor and reuse when possible

**Expected Speedup**: 1-3% reduction in tensor creation overhead

### 8. **Reduce Redundant Softmax Computations** (Low Impact)
**Location**: `src/chess_policy/mcts.py` (expand_node, mcts_search)

**Issue**: Softmax computed multiple times on same logits (once for priors, once for policy distribution)

**Optimizations**:
- Cache softmax result if logits haven't changed
- Only recompute when needed

**Expected Speedup**: <1% but helps with code clarity

## Accuracy Optimizations

### 1. **Better Move Ordering in MCTS** (Medium Impact)
**Location**: `src/chess_policy/mcts.py` (select_child, expand_node)

**Issue**: No move ordering heuristic (MVV-LVA) before PUCT selection. Tactical moves might be explored too late.

**Optimizations**:
- Sort legal moves by MVV-LVA (Most Valuable Victim - Least Valuable Attacker) before creating children
- Prioritize checks, captures, promotions in initial exploration
- This helps MCTS find tactical sequences faster

**Expected Improvement**: 5-10% better tactical play, especially in complex positions

### 2. **Improve Policy Extraction for Underpromotions** (Low-Medium Impact)
**Location**: `src/chess_policy/mcts.py` (expand_node lines 1064-1078)

**Issue**: When multiple indices map to same move (underpromotions), current code takes max but doesn't handle all cases optimally.

**Optimizations**:
- Explicitly handle underpromotion cases
- Consider sum of probabilities for underpromotions rather than max
- Better handling of queen promotions vs underpromotions

**Expected Improvement**: Slightly better endgame play, especially in promotion positions

### 3. **Better Value Head Calibration** (Low Impact)
**Location**: Model architecture, value usage

**Issue**: No explicit calibration of value predictions. Values might be miscalibrated.

**Optimizations**:
- Add value calibration layer (linear transformation) if values are systematically off
- Monitor value accuracy vs actual game outcomes
- Consider temperature scaling for value predictions

**Expected Improvement**: Better position evaluation, especially in endgames

### 4. **Improve Terminal Position Handling** (Low Impact)
**Location**: `src/chess_policy/mcts.py` (expand_node)

**Issue**: Terminal handling is good, but could be more efficient in detecting draws.

**Optimizations**:
- Cache terminal checks (already done with is_terminal flag)
- Early detection of threefold repetition without full board comparison
- Better handling of insufficient material draws

**Expected Improvement**: Slightly better endgame accuracy

### 5. **Better Policy-Value Blending** (Medium Impact)
**Location**: `src/chess_policy/mcts.py` (mcts_search lines 1631-1680)

**Issue**: Current blending (alpha=0.1) and override logic could be improved.

**Optimizations**:
- Adaptive alpha based on search quality (more policy weight when search is noisy)
- Better threshold for policy override (currently 0.8 ratio, could be adaptive)
- Consider visit count confidence in override decision

**Expected Improvement**: 3-7% better move selection, especially when search is limited

## Implementation Priority

### High Priority (Implement First):
1. Reduce redundant board copies (use push/pop)
2. Cache legal moves and legal mask
3. Eliminate redundant policy re-evaluation
4. Improve batch evaluator efficiency

### Medium Priority:
5. Optimize move-to-index mapping
6. Better move ordering (MVV-LVA)
7. Cache FEN strings
8. Better policy-value blending

### Low Priority (Nice to Have):
9. Optimize legal mask tensor creation
10. Reduce redundant softmax computations
11. Improve policy extraction for underpromotions
12. Better value head calibration

## Estimated Overall Impact

**Speed**: 30-50% faster move generation with high-priority optimizations
**Accuracy**: 5-10% improvement in move quality with accuracy optimizations

## Code Quality Improvements

1. **Add type hints** where missing (especially in mcts.py)
2. **Extract constants** for magic numbers (thresholds, batch sizes)
3. **Add docstrings** for optimization-related functions
4. **Profile before/after** to measure actual improvements

