# GEAK-Agent Architecture

## Overview

GEAK-Agent is an **LLM-based multi-agent framework for automatically generating functional and efficient GPU kernels** (specifically Triton kernels for AMD ROCm). It uses an evolutionary genetic algorithm approach with four main components working together to iteratively improve kernel performance.

## Core Components

### 1. Generator
Creates code solutions based on problem descriptions and context.

### 2. Evaluator
Tests functionality and performance through cascaded evaluation:
- Runnable test (syntax/execution)
- Correctness test (output validation)
- Performance test (speedup measurement)

### 3. Reflector
Analyzes failures and successes to provide insights for improvement.

### 4. Optimizer
Improves performance iteratively using evolutionary strategies.

---

## Architecture Flow

```
Problem → Retrieve oneshot → Generate code → 
→ LLM evaluate → Test on GPU → Reflect → 
→ Update candidates → Next iteration
```

---

## Memory System

Each problem maintains a `Memory` object that tracks:

| Field | Description |
|-------|-------------|
| `ps` (ProblemState) | Problem description, instruction, golden reference code |
| `oneshot` | Example code from BM25 retrieval corpus |
| `function_signatures` | Required function signatures for correctness |
| `perf_candidates` | List of top-performing codes (sorted by speedup) |
| `raw_codes` | Current generation's candidate codes |
| `history` | Evolution history per descendant (max 5 attempts) |
| `call_err_msg` / `exe_err_msg` | Error messages from testing |
| `reflection` | Analysis of code quality and issues |
| Status flags | `pass_call`, `pass_exe`, `pass_perf` |

---

## Main Iteration Loop

For each iteration, the agent performs the following steps:

### Step A: Generate Solutions (`generate_solution`)

**Input:** Problem instruction + context  
**Output:** `descendant_num` candidate codes + strategies

**Behavior varies based on state:**

#### Initial Generation
- Uses one-shot example from BM25 corpus
- Provides basic problem context

#### Has Performance Candidates (Optimization Phase)
- Shows top `ancestor_num` performing codes with their speedups
- LLM analyzes patterns and generates improved version
- Can use mutation mode for exploring different strategies
- Includes profiling data if available

#### Debugging Mode
- Uses history to avoid repeating mistakes
- Incorporates error messages and reflections
- Retrieves different one-shot examples

**Prompt includes:**
- Original instruction
- Required function signatures (critical for correctness)
- Example code or performance references
- Optimization strategies:
  - Memory access efficiency
  - Hardware resource utilization
  - IR/Assembly analysis
  - Kernel occupancy
  - Autotuning configurations
  - TorchInductor integration

**Output format:** JSON with `{"strategy": "...", "code": "..."}`

---

### Step B: LLM Evaluation (`generate_llm_evaluate`)

**Input:** Generated code  
**Output:** Numeric quality score (0.0-1.0) across 9 criteria

**Evaluation Criteria:**
1. **Fusion Intelligence** - Smart operation fusion to reduce memory I/O
2. **Autotuning Coverage** - Proper use of `@triton.autotune` with meaningful ranges
3. **Memory Access Efficiency** - Optimized layout and coalesced access
4. **Algorithmic Complexity** - Loop fusion and reduced redundancy
5. **Warp/Wavefront Utilization** - Full compute unit utilization
6. **Software Pipelining** - Appropriate `num_stages` configuration (1-16 for MI250)
7. **Numerical Stability** - Safe operations for large input ranges
8. **Correctness and Portability** - Edge case handling
9. **Optimization Scope** - Use of advanced techniques (e.g., online softmax)

Codes are ranked by total score and stored in `history` for each descendant.

---

### Step C: Run Scripts (Hardware Testing)

**Input:** Generated code  
**Output:** `pass_call`, `pass_exe`, `speedup`, error messages

**Cascaded Evaluation:**

1. **Runnable Test (`pass_call`)**: Can it execute without syntax/runtime errors?
2. **Correctness Test (`pass_exe`)**: Does output match golden reference (allclose)?
3. **Performance Test (`pass_perf`)**: Does it achieve speedup > 0?

**Optional: ROCm Profiling** (if `profiling=True`)
- Uses `rocprof-compute` for hardware-level metrics
- Analyzes memory bandwidth, compute unit occupancy
- Provides insights for optimization strategies

**Early Stopping Logic:**
- If `descendant_debug` codes pass execution test → proceed to next iteration
- Otherwise, continue debugging current generation (up to `max_perf_debug_num` attempts)

---

### Step D: Generate Reflections (`generate_reflexion`)

**Input:** Code + test results + history  
**Output:** Analysis of what went wrong/right

**Two Types of Reflections:**

#### For Failures
- Diagnose errors (syntax, logic, memory access issues)
- Identify incorrect function signatures
- Point out AMD ROCm compatibility issues
- Suggest specific fixes

#### For Successes
- Summarize optimization strategy used
- Analyze performance characteristics
- Identify potential bottlenecks
- Provide insights for further improvement

Reflections guide future generations to:
- Avoid repeating mistakes
- Build on successful strategies
- Understand trade-offs

---

### Step E: Update Performance Candidates

**Goal:** Maintain top `ancestor_num` codes sorted by speedup

**Selection Criteria:**
- Only codes with `pass_perf=True` enter the candidate pool
- Sorted by speedup (higher is better)
- Creates a "hall of fame" that guides future optimizations

**Final Solution Selection:**
1. If `perf_candidates` exists → use best candidate
2. Else if `exe_candidate` exists → use passing execution code
3. Else if `call_candidate` exists → use runnable code
4. Else → use first generated code

---

## Key Mechanisms

### Evolutionary Strategy

The agent implements a genetic algorithm approach:

- **Population:** `descendant_num` codes generated per iteration
- **Selection:** Top performers (by speedup) become `perf_candidates`
- **Crossover:** LLM analyzes multiple ancestors to combine strategies
- **Mutation:** Optional mode to explore different optimization directions
- **Fitness:** Speedup compared to golden reference implementation

### Memory-Guided Learning

Prevents repeating mistakes through historical tracking:

- Each descendant maintains its own `history` (max 5 attempts)
- History includes:
  - Generated code
  - Test results (pass_call, pass_exe, speedup)
  - Reflections and analysis
  - LLM quality scores
- Top 5 attempts preserved based on LLM metrics
- Used in prompts to show evolution trajectory

### Multi-Level Debugging

Sophisticated error recovery mechanism:

1. Code fails → increment `perf_debug_num`
2. If `perf_debug_num < max_perf_debug_num`:
   - Retry with different one-shot example
   - Include error messages and reflections
   - Use historical context
3. If limit exceeded:
   - Reset counter
   - Generate completely new solution
   - Start fresh evolution path

### AMD ROCm Optimization Focus

The system targets AMD GPU-specific optimizations:

**Autotuning Parameters:**
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K` - Tile sizes for tensor operations
- `num_stages` - Pipeline depth (1-16 for MI250)
- `num_warps` - Warps per block (1-16 valid range)
- `GROUP_SIZE_M` - Block grouping strategy

**Performance Strategies:**
- Memory coalescing and layout optimization
- Software pipelining for latency hiding
- Operator fusion (e.g., fused attention, online softmax)
- Shared memory utilization
- Register pressure management
- Grid/block configuration optimization

**ROCm-Specific:**
- Avoid CUDA-specific functions (e.g., `tl.libdevice`)
- Use `tl.math` functions for AMD compatibility
- Target MI200/MI250 GPU architecture
- Respect hardware constraints (num_stages, num_warps limits)

---

## Class Hierarchy

```
BaseAgent
  └── SequentialBaseAgent
        └── Reflexion
              └── Reflexion_Oneshot
                    └── GaAgent
```

### BaseAgent
- Basic agent structure
- Memory initialization
- Single-pass execution
- Multi-threading support

### SequentialBaseAgent
- Adds iteration loop
- Per-iteration checkpointing

### Reflexion
- Adds reflection mechanism
- Error analysis capabilities

### Reflexion_Oneshot
- Adds BM25 retrieval system
- One-shot example integration
- Code retriever for debugging

### GaAgent (Genetic Algorithm Agent)
- Full evolutionary approach
- Performance candidate management
- LLM evaluation system
- Multi-descendant generation
- History tracking per descendant
- Advanced optimization strategies

---

## Configuration Parameters

Key parameters from `tritonbench_gaagent_config.yaml`:

### LLM Configuration
- `api_key`: Authentication for LLM service
- `model_id`: Model to use (GPT, Claude, Gemini)
- `temperature`: Sampling temperature (0.0-2.0)

### Dataset Configuration
- `statis_path`: Problem statistics and metadata
- `py_folder`: Test script folder
- `instruction_path`: Problem instructions
- `corpus_path`: One-shot example corpus
- `golden_metrics`: Reference performance data

### Agent Configuration
- `max_iteration`: Number of evolution cycles
- `ancestor_num`: How many top codes to show as examples (default: 5)
- `descendant_num`: How many codes to generate per iteration (default: 8)
- `descendant_debug`: Minimum passing codes before progressing (default: 1)
- `max_perf_debug_num`: Max debugging attempts before reset (default: 5)
- `gpu_id`: Which GPU to use for testing
- `target_gpu`: Target architecture (MI200/MI250)
- `profiling`: Enable ROCm hardware profiling
- `multi_thread`: Parallel LLM generation

### Checkpointing
- `result_path`: Path to previous results (for resuming)
- `mem_file`: Path to previous memories (for resuming)
- `start_iter`: Starting iteration number
- `start_idx`: Starting problem index
- `output_path`: Output file path

---

## Data Flow Example

### Iteration 0 (Initial Generation)
```
1. Load problem: "Implement fused softmax kernel"
2. Retrieve oneshot: BM25 finds similar softmax example
3. Generate: LLM creates 8 candidate codes
4. LLM Evaluate: Score each on 9 criteria
5. Test: Run on GPU
   - Code 1: pass_call=True, pass_exe=True, speedup=1.2x
   - Code 2: pass_call=True, pass_exe=False (wrong output)
   - Code 3: pass_call=False (syntax error)
   - ...
6. Reflect: Analyze successes and failures
7. Update candidates: Code 1 enters perf_candidates
8. Save: checkpoint_0.jsonl, mem_0.json
```

### Iteration 1 (Optimization)
```
1. Generate: Show Code 1 (1.2x) as ancestor
   - LLM analyzes: "Used basic softmax, can try online softmax"
   - Creates 8 new codes with different strategies
2. LLM Evaluate: Score new codes
3. Test: Run on GPU
   - Code 9: pass_call=True, pass_exe=True, speedup=1.8x
   - Code 10: pass_call=True, pass_exe=True, speedup=1.5x
   - ...
4. Reflect: "Online softmax improved memory access patterns"
5. Update candidates: [Code 9 (1.8x), Code 10 (1.5x), Code 1 (1.2x)]
6. Save: checkpoint_1.jsonl, mem_1.json
```

### Iteration N (Convergence)
```
Eventually converges when:
- New generations don't improve speedup
- All optimization strategies explored
- User-defined max_iteration reached
```

---

## Supported Backends

### LLM Models
- **OpenAI**: GPT-3.5, GPT-4, GPT-5 (via Azure/OpenAI API)
- **Claude**: Claude 2, Claude 3 (via Anthropic API)
- **Gemini**: Gemini Pro (via Google API)

### Datasets
- **TritonBench**: Triton kernel benchmarks
- **ROCm**: ROCm-specific benchmarks
- **Custom**: User-defined datasets (implement `YourData` class)

### Target Hardware
- **AMD GPUs**: MI200, MI250 series
- **Software Stack**: ROCm 5.x+, Triton 3.1.0+

---

## Extending GEAK-Agent

### Adding Custom Dataset

1. Create `dataloaders/YourData.py`
2. Define `YourData` class with required methods:
   ```python
   class YourData:
       def __len__(self) -> int:
           """Return number of problems"""
       
       def load_ps(self, path) -> List[ProblemState]:
           """Load problem states"""
       
       def test_opt_correctness(self, code, filename, tmp_dir, exe_dir):
           """Test code correctness and performance"""
           return pass_call, pass_exe, speedup, stdout, stderr
   ```

3. Update config to use your dataset
4. Run agent: `python main_gaagent.py`

### Adding Custom Agent

1. Create `agents/YourAgent.py`
2. Inherit from appropriate base class
3. Override methods as needed:
   - `memory_init()`: Custom memory structure
   - `generate_solution()`: Custom generation logic
   - `generate_reflexion()`: Custom reflection logic
4. Update main script to use your agent

---

## Output Format

### Result File (.jsonl)
Each line is a JSON object:
```json
{
  "instruction": "Problem description",
  "label": "Golden reference code",
  "filename": "test_softmax.py",
  "predict": "Generated solution code",
  "speedup": 1.8,
  "test_code": "Unit test code"
}
```

### Memory File (.json)
Dictionary mapping filenames to memory states:
```json
{
  "test_softmax.py": {
    "call_err_msg": "error message or None",
    "exe_err_msg": "error message or None",
    "reflection": "Analysis of code",
    "oneshot": "Example code",
    "perf_candidates": [
      ["code", 1.8, 0.0, "strategy", "profiling"],
      ["code", 1.5, 0.0, "strategy", "profiling"]
    ],
    "pass_call": true,
    "pass_exe": true,
    "pass_perf": true
  }
}
```

---

## Resuming from Checkpoints

To resume from a previous run:

```yaml
# In config file
result_path: "../outputs/optimagent_10.jsonl"
mem_file: "../outputs/optimagent_mem_10.json"
start_iter: 11
```

This allows:
- Continuing optimization from any iteration
- Recovering from interruptions
- Incremental improvement experiments
- A/B testing different strategies

---

## Key Design Principles

### 1. Correctness First
- Always validate function signatures
- Run comprehensive tests before performance tuning
- Never sacrifice correctness for speed

### 2. Evolutionary Improvement
- Start with working baseline
- Small iterative improvements
- Learn from both successes and failures

### 3. Memory-Guided Learning
- Track historical attempts
- Avoid repeating mistakes
- Build on successful patterns

### 4. Hardware-Aware Optimization
- Target specific GPU architecture
- Use profiling data when available
- Respect hardware constraints

### 5. Explainable AI
- Generate strategies with code
- Provide reflections on failures
- Track optimization reasoning

---

## Performance Considerations

### Parallelization
- LLM generation: multi-threaded (3 workers default)
- GPU testing: sequential (to avoid resource conflicts)
- Multiple problems: parallel processing supported

### Resource Usage
- LLM API costs: ~30K tokens per generation
- GPU memory: varies by kernel complexity
- Disk space: checkpoints can be large for many iterations

### Bottlenecks
1. **LLM latency**: Most time spent waiting for LLM responses
2. **GPU testing**: Kernel compilation and execution time
3. **Profiling**: Hardware profiling adds significant overhead

---

## Future Directions

Potential improvements and research directions:

1. **Multi-objective optimization**: Balance latency, efficiency, and power
2. **Transfer learning**: Apply knowledge across problem domains
3. **Ensemble methods**: Combine multiple LLM strategies
4. **Formal verification**: Prove correctness guarantees
5. **Auto-scaling**: Adapt to different problem sizes
6. **Cross-platform**: Support NVIDIA, Intel GPUs

---

## References

- Triton Documentation: https://triton-lang.org/
- ROCm Documentation: https://rocm.docs.amd.com/
- GEAK-eval: Evaluation framework for GPU kernels
- TritonBench: Benchmark suite for Triton kernels

---

## License

See LICENSE.md in the repository root.
