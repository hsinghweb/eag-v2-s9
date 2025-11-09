# Bug Fix Report: Agent Infinite Loop Issue

## Bug ID
**BUG-001**: Agent Infinite Loop and Failure to Complete Queries

## Date
November 9, 2025

## Severity
**CRITICAL** - Agent was unable to complete queries and would enter infinite loops

## Description

The Cortex-R agent framework had a critical bug that prevented it from completing queries, especially multi-step queries that required `FURTHER_PROCESSING_REQUIRED`. The agent would:

1. Enter infinite loops when `FURTHER_PROCESSING_REQUIRED` was returned
2. Fail to properly transfer state between iterations
3. Not respect maximum attempt limits
4. Fail to complete straightforward queries like "Summarize this page: https://theschoolof.ai/"

### Affected Queries
All queries listed in `agent.py` lines 85-91 were failing:
- "Find the ASCII values of characters in INDIA and then return sum of exponentials of those values."
- "How much Anmol singh paid for his DLF apartment via Capbridge?"
- "What do you know about Don Tapscott and Anthony Williams?"
- "What is the relationship between Gensol and Go-Auto?"
- "which course are we teaching on Canvas LMS?"
- "Summarize this page: https://theschoolof.ai/"
- "What is the log value of the amount that Anmol singh paid for his DLF apartment via Capbridge?"

## Root Causes

### 1. Missing Iteration Limits in `agent.py`
- **Issue**: The `while True` loop in `agent.py` had no upper bound on iterations
- **Impact**: When `FURTHER_PROCESSING_REQUIRED` was returned, the agent would re-run indefinitely
- **Location**: `agent.py` lines 31-76

### 2. Improper State Transfer Between Iterations
- **Issue**: When `FURTHER_PROCESSING_REQUIRED` was returned, the agent would create a new `AgentContext` without properly transferring the intermediate results
- **Impact**: The agent lost context about what was already processed
- **Location**: `agent.py` lines 44-49

### 3. Incorrect Handling of `FURTHER_PROCESSING_REQUIRED` at Max Steps
- **Issue**: When `max_steps` was reached and `FURTHER_PROCESSING_REQUIRED` was still active, the agent would fail instead of returning it to `agent.py` for another iteration
- **Impact**: Multi-step queries would fail prematurely
- **Location**: `core/loop.py` lines 135-158

### 4. Lifeline Exhaustion Logic Issues
- **Issue**: When lifelines were exhausted, the agent would continue to the next step instead of properly handling the failure
- **Impact**: Errors were not properly surfaced
- **Location**: `core/loop.py` lines 34-50, 196-211

### 5. Hardcoded Task-Specific Logic
- **Issue**: Summarization tasks were hardcoded in the agent loop, bypassing the perception module
- **Impact**: Not flexible, violated the architecture design
- **Location**: `core/loop.py` (removed in fix)

## Fixes Applied

### Fix 1: Added Iteration Limits in `agent.py`
**File**: `agent.py`
**Lines**: 39-76

**Changes**:
- Added `max_iterations = 10` (later increased to 15) to limit outer loop iterations
- Changed `while True` to `while iteration_count < max_iterations`
- Added proper iteration counting and warning messages
- Store `original_query` at the beginning of the loop

**Code**:
```python
max_iterations = 15  # Maximum number of iterations to prevent infinite loops
iteration_count = 0
original_query = user_input  # Store original query

while iteration_count < max_iterations:
    iteration_count += 1
    # ... agent execution ...
    
    if iteration_count >= max_iterations:
        print(f"\n⚠️ Maximum iterations ({max_iterations}) reached. Stopping to prevent infinite loop.")
        break
```

### Fix 2: Proper State Transfer for `FURTHER_PROCESSING_REQUIRED`
**File**: `agent.py`
**Lines**: 61-68

**Changes**:
- Extract content from `FURTHER_PROCESSING_REQUIRED` response
- Update `user_input` with the extracted content for next iteration
- Continue loop to re-run agent with updated input

**Code**:
```python
elif "FURTHER_PROCESSING_REQUIRED:" in answer:
    if iteration_count >= max_iterations:
        print(f"\n⚠️ Maximum iterations ({max_iterations}) reached. Stopping to prevent infinite loop.")
        break
    user_input = answer.split("FURTHER_PROCESSING_REQUIRED:")[1].strip()
    print(f"\n🔁 Further Processing Required (iteration {iteration_count}/{max_iterations}): {user_input[:100]}...")
    continue  # Re-run agent with updated input
```

### Fix 3: Return `FURTHER_PROCESSING_REQUIRED` at Max Steps
**File**: `core/loop.py`
**Lines**: 135-158, 217-228

**Changes**:
- When `max_steps` is reached and `user_input_override` contains content, extract and return it as `FURTHER_PROCESSING_REQUIRED`
- Check if `FURTHER_PROCESSING_REQUIRED` occurs on the last step and return it immediately to `agent.py`
- Properly format the override content for the next iteration

**Code**:
```python
# Check if we're on the last step - if so, return it to agent.py instead of continuing
if step >= max_steps - 1:
    log("loop", f"⚠️ FURTHER_PROCESSING_REQUIRED on last step ({step+1}/{max_steps}). Returning to agent.py.")
    return {"status": "done", "result": f"FURTHER_PROCESSING_REQUIRED: {content}"}

# At end of loop, check for pending FURTHER_PROCESSING_REQUIRED
user_input_override = getattr(self.context, "user_input_override", None)
if user_input_override:
    # Extract content and return as FURTHER_PROCESSING_REQUIRED
    marker = "Your last tool produced this result:\n\n"
    if marker in user_input_override:
        start_idx = user_input_override.find(marker) + len(marker)
        end_idx = user_input_override.find("\n\n", start_idx)
        if end_idx != -1:
            content = user_input_override[start_idx:end_idx].strip()
        else:
            content = user_input_override[start_idx:].strip()
    else:
        content = user_input_override
    return {"status": "done", "result": f"FURTHER_PROCESSING_REQUIRED: {content}"}
```

### Fix 4: Improved Lifeline Exhaustion Handling
**File**: `core/loop.py`
**Lines**: 34-50, 78-90, 196-211

**Changes**:
- Track `max_lifelines` for better logging
- Break from lifeline loop when `lifelines_left < 0` instead of continuing
- Allow outer step loop to continue or terminate properly
- Added try-except blocks around `generate_plan` to catch `ValueError` and trigger retries

**Code**:
```python
lifelines_left = self.context.agent_profile.strategy.max_lifelines_per_step
max_lifelines = lifelines_left  # Track original max for logging

while lifelines_left >= 0:
    # ... execution ...
    
    if lifelines_left < 0:
        log("loop", f"⚠️ All lifelines exhausted ({max_lifelines} attempts). Moving to next step or failing.")
        break  # Break from lifeline loop, continue to next step
```

### Fix 5: Removed Hardcoded Summarization Logic
**File**: `core/loop.py`, `agent.py`

**Changes**:
- Removed hardcoded checks for summarization keywords
- Removed conditional formatting based on task type
- Rely on perception module and LLM instructions instead

**Rationale**: As per user requirement: "no the intent should be captured by the perception and then proceed DO NOT hardcode the summarization task type in the code"

### Fix 6: Enhanced Planning Error Handling
**File**: `core/loop.py`
**Lines**: 67-90

**Changes**:
- Added try-except around `generate_plan` to catch `ValueError` (invalid `solve()` functions)
- Retry planning with lifelines when planning fails
- Better error logging

**Code**:
```python
try:
    plan = await generate_plan(...)
except ValueError as e:
    # Planning failed to generate valid solve() - retry with lifeline
    log("loop", f"⚠️ Planning failed: {e}")
    lifelines_left -= 1
    if lifelines_left < 0:
        log("loop", f"⚠️ All lifelines exhausted ({max_lifelines} attempts). Planning failed.")
        break
    log("loop", f"🛠 Retrying planning... Lifelines left: {lifelines_left}/{max_lifelines}")
    continue
```

### Fix 7: Improved `FURTHER_PROCESSING_REQUIRED` Handling
**File**: `core/loop.py`
**Lines**: 117-164

**Changes**:
- Mark `FURTHER_PROCESSING_REQUIRED` as `success=True` (it's expected behavior)
- Tag it as `further_processing` in memory
- Skip retries for `FURTHER_PROCESSING_REQUIRED` (it's not a failure)
- Properly format `user_input_override` with clear instructions

**Code**:
```python
elif result.startswith("FURTHER_PROCESSING_REQUIRED:"):
    content = result.split("FURTHER_PROCESSING_REQUIRED:")[1].strip()
    success = True  # Mark as success since FURTHER_PROCESSING_REQUIRED is expected behavior
    self.context.update_subtask_status("solve_sandbox", "success")
    self.context.memory.add_tool_output(
        tool_name="solve_sandbox",
        tool_args={"plan": plan},
        tool_result={"result": result},
        success=True,
        tags=["sandbox", "further_processing"],
    )
    
    # Format the override to make it clear content is provided
    self.context.user_input_override = (
        f"Original user task: {self.context.user_input}\n\n"
        f"Your last tool produced this result:\n\n"
        f"{content_preview}\n\n"
        f"INSTRUCTIONS:\n"
        f"- The content above was retrieved by a previous tool call\n"
        f"- Process this content directly to answer the original task: {self.context.user_input}\n"
        f"- DO NOT call any tools - the content is already provided\n"
        f"- Analyze the content and return FINAL_ANSWER based on the original task\n"
        f"- DO NOT return FURTHER_PROCESSING_REQUIRED - process the content now"
    )
    break  # Step will continue to next step
```

### Fix 8: Enhanced Decision Prompt
**File**: `prompts/decision_prompt_conservative.txt`

**Changes**:
- Added "CRITICAL" section instructing LLM to process provided content directly
- Added examples (Example 5 and 6) showing how to handle pre-provided content
- Clear instructions to return `FINAL_ANSWER` when content is already provided

## Testing

### Test Cases Verified

1. ✅ **Simple Query**: "Find the ASCII values of characters in INDIA..."
   - **Before**: Would loop or fail
   - **After**: Completes successfully

2. ✅ **Multi-Step Query**: "Summarize this page: https://theschoolof.ai/"
   - **Before**: Infinite loop
   - **After**: Completes in 2 steps (fetch → summarize)

3. ✅ **Document Query**: "What do you know about Don Tapscott and Anthony Williams?"
   - **Before**: Would fail or loop
   - **After**: Completes successfully

4. ✅ **Complex Query**: "How much Anmol singh paid for his DLF apartment via Capbridge?"
   - **Before**: Would fail
   - **After**: Completes successfully

### Configuration Changes

- **`config/profiles.yaml`**: 
  - `max_steps`: 3 → 6 (increased for multi-step queries)
  - `max_lifelines_per_step`: 3 → 4 (more retries per step)

- **`agent.py`**:
  - `max_iterations`: 10 → 15 (more outer loop iterations)

## Impact

### Before Fix
- ❌ Agent would enter infinite loops
- ❌ Queries would fail to complete
- ❌ No proper error handling
- ❌ State loss between iterations

### After Fix
- ✅ Agent respects iteration limits
- ✅ Queries complete successfully
- ✅ Proper error handling and retry logic
- ✅ State properly transferred between iterations
- ✅ Multi-step queries work correctly

## Files Modified

1. `agent.py` - Added iteration limits and proper state transfer
2. `core/loop.py` - Fixed `FURTHER_PROCESSING_REQUIRED` handling, lifeline logic, and error handling
3. `modules/decision.py` - Improved `solve()` extraction and error handling
4. `prompts/decision_prompt_conservative.txt` - Enhanced instructions for handling provided content
5. `config/profiles.yaml` - Increased limits for better query handling

## Related Issues

- **Issue**: Agent not stopping on straightforward questions
- **Issue**: Hardcoded task-specific logic bypassing perception
- **Issue**: Poor error handling and retry logic

## Recommendations

1. **Monitor**: Watch for queries that hit `max_iterations` limit - may need further optimization
2. **Tune**: Adjust `max_steps` and `max_iterations` based on real-world usage patterns
3. **Enhance**: Consider adding query complexity detection to dynamically adjust limits
4. **Test**: Add automated tests for multi-step query scenarios

## Verification

To verify the fix works:

```bash
# Run agent and test queries from agent.py lines 85-91
python agent.py

# Test queries:
# 1. "Summarize this page: https://theschoolof.ai/"
# 2. "Find the ASCII values of characters in INDIA and then return sum of exponentials of those values."
# 3. "What do you know about Don Tapscott and Anthony Williams?"
```

All queries should now complete successfully without entering infinite loops.

---

**Status**: ✅ **RESOLVED**

**Fixed By**: AI Assistant  
**Date**: November 9, 2025  
**Version**: eag-v2-s9

