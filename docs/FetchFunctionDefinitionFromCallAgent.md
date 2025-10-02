# FetchFunctionDefinitionFromCallAgent Update Plan

This document captures the planned refactor for `FetchFunctionDefinitionFromCallAgent`
and the supporting helper queries. No code has been changed yet.

## Objectives
- Align the agent with the enhanced `ExtractMemberDefinitionFromClassQuery` prompt
  so inline definitions (constructors, destructors, special members) can be
  resolved reliably.
- Introduce a single decision flow that covers the major call-expression
  variants discovered during recent debugging.
- Provide reusable helpers and diagnostics that future agents can consume.

## Inputs & Dependencies
- `AnalyzeCallExpressionPipeline`: supplies call classification, qualified names,
  template arguments, and language hints.
- `FetchClassPipeline`: retrieves complete class definitions when inline
  members must be inspected.
- `FindCallOperatorQuery`: extracts callable operator bodies.
- `ExtractMemberDefinitionFromClassQuery`: now supports both targeted and bulk
  extraction with `kind` filters.

## Planned Workflow
1. **Normalize Inputs**
   - Trim `callee_name`, `callee_expr`, and caller body.
   - Capture an analysis snapshot (`self._last_analysis`) after invoking
     `AnalyzeCallExpressionPipeline` for observability.

2. **Branching Logic**
   - `simple_function` / `namespaced_function`: proceed directly to retrieval
     with provided qualified names.
   - `member_method`: when `needs_class_definition` is true, call
     `ExtractMemberDefinitionFromClassQuery` with `kind="member"`; otherwise
     fall back to retrieval using `Class::method` variants.
   - `static_method`: treat similarly to member but skip inline extraction unless
     explicitly requested by the analysis.
   - `call_operator`: fetch class, attempt `FindCallOperatorQuery`, then fall
     back to `[FUNCDEF]Class::operator()` retrieval keywords.
   - `constructor` / `destructor`: use the refreshed extractor with
     `kind="constructor"` / `kind="destructor"`, then add common mangled name
     patterns to the candidate list.
   - `template_function`: merge template arguments and raw call expression into
     the keyword set to increase recall.
   - `function_pointer` / `macro`: emit debug log and fall back to generic search
     (future work could add dedicated pipelines).

3. **Retrieval Helper Enhancements**
   - Deduplicate BM25 keywords and record the current terms used (for logging).
   - Keep using `OUTPUT_FORMAT_TEMPLATE` to enforce complete definitions; on
     failure, fall back to the first code block in the LLM reply.
   - Maintain the existing iteration over retrieved documents, respecting
     `COUNT_UPPER_LIMIT` and `CALL_CHAIN_WIDTH` (no change needed).

4. **Result Aggregation**
   - Gather definitions from specialist handlers and generic fallbacks, then run
     `deduplicate_strings` before returning.
   - Surface empty results cleanly so higher-level caching continues to work.

5. **Follow-up Tasks (post code-change)**
   - Update or add unit tests for member/constructor/destructor flows once the
     agent changes are in place.
   - Document the new `_last_analysis` attribute for debugging utilities.
   - Consider splitting shared helpers into a separate module if additional
     agents start reusing them.

This plan should be reviewed before implementing the corresponding code
updates in `ragalyze/agent.py`.
