# 🧩 Heuristics That Run on Queries and Results

Here are **10 practical Heuristics** that can run on both **user queries** and **LLM/tool results** — designed to keep your Agent safe, robust, and precise before/after reasoning takes place.  

---

## 🧠 Input (Query) Heuristics
These run **before** sending anything to the LLM or tools.

### 1. Banned Word Filter
- ✅ Block or sanitize input containing banned, offensive, or unsafe words (e.g., violence, self-harm, explicit terms).  
- 💡 **Purpose:** Prevent prompt injection or unsafe responses.

### 2. Tool Invocation Whitelist
- ✅ Allow only registered tool names (e.g., `search_web`, `run_python`) — reject unrecognized or hidden tool calls.  
- 💡 **Purpose:** Prevent unauthorized or harmful tool execution.

### 3. Prompt Injection Detector
- ✅ Look for suspicious patterns like `"ignore previous instructions"` or `"print system prompt"`.  
- 💡 **Purpose:** Protect internal logic and secrets from user manipulation.

### 4. Input Length & Format Checker
- ✅ Reject or truncate excessively long inputs, malformed JSON, or invalid parameter formats.  
- 💡 **Purpose:** Protect system resources and ensure structured input.

### 5. Safety Context Rule
- ✅ If query requests action on personal data, URLs, or external APIs — require user confirmation or permission flag.  
- 💡 **Purpose:** Enforce consent and data safety.

---

## ⚙️ Output (Result) Heuristics
These run **after** receiving results from LLM or tools.

### 6. Sensitive Data Scrubber
- ✅ Detect and redact phone numbers, emails, or secrets (API keys, tokens) before showing to user.  
- 💡 **Purpose:** Prevent data leaks or exposure.

### 7. Result Sanity Checker
- ✅ Verify expected structure: if expecting JSON, ensure it’s valid; if expecting a list, ensure proper types.  
- 💡 **Purpose:** Prevent downstream errors in multi-step workflows.

### 8. Hallucination Detector (Keyword Match)
- ✅ Compare entities in output with context or database; flag mismatches like “nonexistent tool names.”  
- 💡 **Purpose:** Reduce misinformation from LLM outputs.

### 9. Content Category Filter
- ✅ Use keyword heuristics or regex to classify and block unsafe content (e.g., medical advice without disclaimer).  
- 💡 **Purpose:** Maintain compliance and ethical boundaries.

### 10. Timeout & Retry Policy
- ✅ If tool response is delayed or malformed, retry up to 3 times or fallback to LLM summary.  
- 💡 **Purpose:** Ensure reliability and graceful failure handling.

---

## 💬 Summary

| Category | Examples | Purpose |
|-----------|-----------|----------|
| **Input Heuristics** | Banned words, Tool whitelist, Format checker | Guardrails before reasoning |
| **Output Heuristics** | Data scrubber, Hallucination check, Retry policy | Sanity checks after reasoning |

---

> 🧩 **Key Idea:**  
> LLMs understand *intent*.  
> Heuristics enforce *safety and structure*.  
> Together, they make an Agent reliable and intelligent.
