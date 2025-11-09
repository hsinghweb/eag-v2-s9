from typing import List, Optional
from modules.perception import PerceptionResult
from modules.memory import MemoryItem
from modules.model_manager import ModelManager
from modules.tools import load_prompt
from modules.conversation_indexer import get_conversation_indexer
import re

# Optional logging fallback
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

model = ModelManager()


# prompt_path = "prompts/decision_prompt.txt"

async def generate_plan(
    user_input: str, 
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str],
    prompt_path: str,
    step_num: int = 1,
    max_steps: int = 3,
) -> str:

    """Generates the full solve() function plan for the agent."""

    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"

    prompt_template = load_prompt(prompt_path)
    
    # Get relevant historical conversations
    historical_context = ""
    try:
        indexer = get_conversation_indexer()
        # Only search if user_input looks like a new query (not already processed content)
        if "Your last tool produced this result:" not in user_input:
            historical_context = indexer.get_relevant_context(user_input, top_k=2)
            if historical_context:
                log("plan", f"📚 Found {len(historical_context.split('Q:')) - 1} relevant past conversations")
    except Exception as e:
        log("plan", f"⚠️ Error getting historical context: {e}")
    
    # Check if user_input contains content from a previous tool (indicated by "Your last tool produced this result:")
    has_provided_content = "Your last tool produced this result:" in user_input or "CONTENT TO" in user_input.upper()
    
    # Add context about processing provided content if detected
    content_instruction = ""
    if has_provided_content:
        content_instruction = "\n\n⚠️ IMPORTANT: The user_input above contains content that was already retrieved by a previous tool. You MUST process this content directly - DO NOT call any tools. Return FINAL_ANSWER based on the original task and the provided content.\n"

    prompt = prompt_template.format(
        tool_descriptions=tool_descriptions,
        user_input=user_input
    )
    
    # Add historical context and content instruction
    if historical_context:
        prompt = f"{prompt}\n\n{historical_context}\n"
    if content_instruction:
        prompt = f"{prompt}{content_instruction}"


    try:
        raw = (await model.generate_text(prompt)).strip()
        log("plan", f"LLM output (first 500 chars): {raw[:500]}...")

        # If fenced in ```python ... ```, extract
        if raw.startswith("```"):
            # Find the closing ```
            end_idx = raw.find("```", 3)
            if end_idx != -1:
                raw = raw[3:end_idx].strip()
                if raw.lower().startswith("python"):
                    raw = raw[len("python"):].strip()
            else:
                # No closing ```, try to extract anyway
                raw = raw.strip("`").strip()
                if raw.lower().startswith("python"):
                    raw = raw[len("python"):].strip()

        # Try to find solve function
        if re.search(r"^\s*(async\s+)?def\s+solve\s*\(", raw, re.MULTILINE):
            return raw  # ✅ Correct, it's a full function
        else:
            # Try to extract solve function even if it's embedded in other text
            # Look for async def solve or def solve
            patterns = [
                r"(async\s+def\s+solve\s*\([^)]*\)\s*:.*?)(?=\n\s*(?:async\s+)?def\s+|\n\s*#|\Z)",
                r"(def\s+solve\s*\([^)]*\)\s*:.*?)(?=\n\s*(?:async\s+)?def\s+|\n\s*#|\Z)",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw, re.DOTALL | re.MULTILINE)
                if match:
                    extracted = match.group(1).strip()
                    # Ensure it starts with def or async def
                    if not extracted.startswith(("def ", "async def ")):
                        # Try to find the actual start
                        def_match = re.search(r"(async\s+)?def\s+solve", extracted)
                        if def_match:
                            start = def_match.start()
                            extracted = extracted[start:].strip()
                    log("plan", "⚠️ Extracted solve() from embedded text")
                    return extracted
            
            # Last resort: try to find anything that looks like a function
            func_match = re.search(r"((?:async\s+)?def\s+\w+\s*\([^)]*\)\s*:.*)", raw, re.DOTALL)
            if func_match:
                extracted = func_match.group(1).strip()
                # Rename function to solve if needed
                if "def solve" not in extracted and "def " in extracted:
                    extracted = re.sub(r"def\s+\w+", "def solve", extracted, count=1)
                log("plan", "⚠️ Extracted and renamed function to solve()")
                return extracted
            
            log("plan", "⚠️ LLM did not return a valid solve(). Will retry with lifeline.")
            # Raise ValueError to trigger retry instead of immediate failure
            raise ValueError("Could not generate valid solve() function from LLM output")

    except ValueError:
        # Re-raise ValueError to trigger retry
        raise
    except Exception as e:
        log("plan", f"⚠️ Planning exception: {e}")
        raise ValueError(f"Planning failed with exception: {str(e)}")
