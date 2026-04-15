---
created: 2026-04-15T13:07:48+08:00
updated: 2026-04-15T17:08:54+08:00
---
# nano-vLLM Learning Plan — Master LLM Inference

## Status
- **Created**: 2025-07-15
- **Current Phase**: Phase 0 (Not started)
- **Last Session**: Initial plan creation

## Learning Repos
- **Primary (performance)**: https://github.com/GeeeekExplorer/nano-vllm (~1200 lines, matches vLLM perf)
- **Educational fork**: https://github.com/ovshake/nano-vllm (narrator/xray/tutorial/dashboard modes)
- **v1 architecture**: https://github.com/slwang-ustc/nano-vllm-v1 (chunked prefill, v1 scheduler)
- **DeepWiki docs**: https://deepwiki.com/GeeeekExplorer/nano-vllm

## Phase Progress Tracker
- [ ] Phase 0: Setup & Mental Model (1 week)
- [ ] Phase 1: Transformers, Tokenization, Attention (2-3 weeks)
- [ ] Phase 2: Single-request Generation, Prefill/Decode, Sampling (2 weeks)
- [ ] Phase 3: KV Cache & PagedAttention (3 weeks)
- [ ] Phase 4: Continuous Batching & Scheduling (3 weeks)
- [ ] Phase 5: Performance Optimization (4 weeks)
- [ ] Phase 6: Tensor Parallelism (2-3 weeks)
- [ ] Phase 7: Benchmarking & Capstone (2-3 weeks)

## Session Log
- **2025-07-15**: Created comprehensive learning plan (synthesized from Claude + Codex research)

## Expected Deliverables
- inference_journal.md
- toy_attention.py
- toy_generate.py
- kv_cache_simulator.py
- scheduler_playground.ipynb
- benchmark_matrix.md
- one capstone extension or PR-quality patch

## Learning Loop (use for every subsystem)
1. Run ovshake/nano-vllm in --tutorial or --narrate
2. Re-run with --xray
3. Read the matching upstream file
4. Reimplement the smallest possible version yourself
5. Benchmark or visualize it
