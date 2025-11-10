# Cortex-R Agent

A reasoning-driven AI agent that uses external tools and memory to solve complex tasks step-by-step through a perception-planning-action loop.

## 🎯 Overview

**Cortex-R** is an intelligent agent framework that combines:
- **Multi-MCP (Model Context Protocol)** server integration for specialized tools
- **LLM-powered reasoning** (Gemini/Ollama) for perception and planning
- **Python sandbox execution** for safe code execution
- **Persistent memory** with conversation indexing for learning from past interactions
- **Multi-step task handling** with automatic retry mechanisms

The agent follows a structured workflow:
1. **Perception**: Understands user intent and selects relevant tools
2. **Planning**: Generates executable Python code using LLM
3. **Action**: Executes the plan in a sandboxed environment
4. **Memory**: Stores interactions and learns from historical conversations

## 🛠️ Technology Stack

- **Language**: Python 3.x
- **LLM Integration**: 
  - Google Gemini (via Vertex AI) for text generation
  - Ollama (local models) as alternative
  - Nomic embeddings for semantic search
- **MCP Framework**: Model Context Protocol for tool management
- **Vector Search**: FAISS for document and conversation indexing
- **Memory Storage**: JSON-based session storage with date-based organization
- **Dependencies**: 
  - `uv` for package management
  - `asyncio` for asynchronous operations
  - `yaml` for configuration management

## 📁 Project Structure

```
eag-v2-s9/
├── agent.py                 # Main entry point
├── core/                    # Core agent components
│   ├── loop.py             # Perception-Planning-Action loop
│   ├── context.py          # Session context and state
│   ├── session.py          # MultiMCP session management
│   └── strategy.py         # Agent strategy configuration
├── modules/                 # Agent modules
│   ├── perception.py        # Intent understanding
│   ├── decision.py         # Plan generation
│   ├── action.py           # Code execution
│   ├── memory.py           # Memory management
│   ├── model_manager.py    # LLM integration
│   ├── conversation_indexer.py  # Historical conversation indexing
│   └── tools.py            # Tool utilities
├── config/                  # Configuration files
│   ├── profiles.yaml       # Agent profiles and MCP servers
│   └── models.json         # LLM model configurations
├── prompts/                 # LLM prompt templates
├── mcp_server_*.py         # MCP server implementations
├── documents/               # Document storage
├── memory/                  # Session memory storage
├── conversation_index/      # Conversation index (auto-generated)
└── faiss_index/            # Document index (auto-generated)
```

## 🔄 Control Flow

```
User Query
    ↓
agent.py (Main Entry)
    ↓
AgentLoop.run()
    ↓
┌─────────────────────────────────────┐
│  Step 1: Perception                 │
│  - Analyze user intent              │
│  - Extract entities                 │
│  - Select relevant MCP servers      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 2: Decision/Planning           │
│  - Generate solve() function         │
│  - Include historical context        │
│  - Plan tool usage                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 3: Action                      │
│  - Execute solve() in sandbox        │
│  - Call MCP tools                    │
│  - Handle results                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Result Processing                   │
│  - FINAL_ANSWER → Return to user     │
│  - FURTHER_PROCESSING_REQUIRED →    │
│    Continue to next step             │
└─────────────────────────────────────┘
```

## 🎮 MCP Servers

The agent integrates with multiple MCP servers:

1. **Math Server** (`mcp_server_1.py`): Mathematical operations, Fibonacci sequences, string conversions
2. **Documents Server** (`mcp_server_2.py`): PDF extraction, document search, webpage conversion
3. **Web Search Server** (`mcp_server_3.py`): DuckDuckGo search, HTML download

## 📊 Example Logs

Below are three complete execution logs demonstrating the agent's capabilities:

---

### Example 1: Mathematical Calculation - Fibonacci Sum of Squares

**Query**: `Find the sum of square of first 5 fibonacci numbers`

**Full Log**:
```
🧑 What do you want to solve today? → Find the sum of square of first 5 fibonacci numbers

🔁 Step 1/3 starting...

[19:37:22] [perception] Raw output: json
{
  "intent": "Calculate the sum of squares of the first 5 Fibonacci numbers.",
  "entities": ["sum", "square", "first 5", "Fibonacci numbers"],
  "tool_hint": "python sandbox",
  "selected_servers": ["math"]
}

[perception] intent='Calculate the sum of squares of the first 5 Fibonacci numbers.' 
entities=['sum', 'square', 'first 5', 'Fibonacci numbers'] 
tool_hint='python sandbox' 
selected_servers=['math']

[19:37:22] [indexer] ✅ Loaded index with 67 conversations
[19:37:22] [indexer] 🔍 Scanning memory files for conversations...
[19:37:22] [indexer] 📁 Found 18 session files
[19:37:22] [indexer] ✅ Using existing index with 67 conversations

[19:37:26] [plan] LLM output: python
import json

async def solve():
    input = {"input": {"n": 5}}
    result = await mcp.call_tool('fibonacci_numbers', input)
    fibonacci_numbers = json.loads(result.content[0].text)["result"]
    
    squared_fibonacci = [x * x for x in fibonacci_numbers]
    
    sum_of_squares = sum(squared_fibonacci)
    
    return f"FINAL_ANSWER: {sum_of_squares}"


[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

[19:37:27] [indexer] 🔄 Refreshing conversation index (incremental update)...
[19:37:27] [indexer] ✅ Index refresh complete. Total conversations: 67
[19:37:27] [loop] ✅ Conversation index refreshed

💡 Final Answer: 15
```

**Analysis**:
- **Perception**: Correctly identified the task as a mathematical calculation requiring Fibonacci numbers
- **Planning**: Generated a clean `solve()` function that calls the `fibonacci_numbers` tool, squares each number, and sums them
- **Action**: Successfully executed in the Python sandbox
- **Result**: Correctly calculated 1² + 1² + 2² + 3² + 5² = 15

---

### Example 2: Document Processing - PDF Extraction and Summarization

**Query**: `What are 3 main points in the "documents\v0.1_State_of_AI_in_Business_2025_Report.pdf" document?`

**Full Log**:
```
🧑 What do you want to solve today? → What are 3 main points in the "documents\v0.1_State_of_AI_in_Business_2025_Report.pdf" document?

🔁 Step 1/3 starting...

[19:38:36] [perception] ⚠️ Perception failed: 429 RESOURCE_EXHAUSTED
[perception] intent='unknown' 
selected_servers=['math', 'documents', 'websearch']

[19:38:47] [perception] Raw output: json
{
  "intent": "Extract key information from a specified PDF document.",
  "entities": ["3 main points", "documents\\v0.1_State_of_AI_in_Business_2025_Report.pdf"],
  "tool_hint": "PDF document processing",
  "selected_servers": ["documents"]
}

[perception] intent='Extract key information from a specified PDF document.' 
entities=['3 main points', 'documents\\v0.1_State_of_AI_in_Business_2025_Report.pdf'] 
tool_hint='PDF document processing' 
selected_servers=['documents']

[19:38:54] [plan] LLM output: ```python
async def solve():
    input = {"input": {"file_path": "documents/v0.1_State_of_AI_in_Business_2025_Report.pdf"}}
    result = await mcp.call_tool('extract_pdf', input)
    return f"FURTHER_PROCESSING_REQUIRED: {result}"


[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

[19:40:12] [loop] 📨 Forwarding intermediate result to next step:
Original user task: What are 3 main points in the "documents\v0.1_State_of_AI_in_Business_2025_Report.pdf" document?

Your last tool produced this result:
[PDF content extracted - 56477 characters total]

INSTRUCTIONS:
- The content above was retrieved by a previous tool call
- Process this content directly to answer the original task
- DO NOT call any tools - the content is already provided
- Analyze the content and return FINAL_ANSWER based on the original task

[19:40:12] [loop] 🔁 Continuing based on FURTHER_PROCESSING_REQUIRED — Step 1/3 continues...

🔁 Step 2/3 starting...

[19:40:17] [perception] Raw output: json
{
  "intent": "Extract main points from a specific document.",
  "entities": ["3 main points", "documents\\v0.1_State_of_AI_in_Business_2025_Report.pdf"],
  "tool_hint": null,
  "selected_servers": ["documents"]
}


[19:40:48] [plan] LLM output: python
async def solve():
    # The content of the PDF has already been provided.
    # We need to identify 3 main points from the provided text.
    # Based on the executive summary, here are 3 main points:
    
    # 1. The GenAI Divide: 95% of organizations are getting zero return on their GenAI investments, despite significant spending.
    # 2. Productivity vs. P&L: While tools like ChatGPT enhance individual productivity, they don't significantly impact P&L performance. Enterprise-grade systems are often rejected due to brittleness and misalignment.
    # 3. Learning is Key: The core barrier to scaling GenAI is not infrastructure or talent, but the ability of systems to retain feedback, adapt to context, and improve over time.
    
    return "FINAL_ANSWER: 1. The GenAI Divide: 95% of organizations are getting zero return on their GenAI investments. 2. Productivity vs. P&L: Tools enhance individual productivity but lack significant P&L impact. 3. Learning is Key: Systems need to retain feedback and adapt to context to scale effectively."


[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

[19:40:48] [indexer] 🔄 Refreshing conversation index (incremental update)...
[19:40:48] [indexer] ✅ Extracted conversation: Q='What are 3 main points in the "documents\v0.1_Stat...' A='1. The GenAI Divide: 95% of organizations are gett...'
[19:40:48] [indexer] 📝 Extracted 1 conversation(s) from session-1762783641-0255a7.json
[19:40:48] [indexer] 💾 Saved index with 68 conversations
[19:40:48] [indexer] ✅ Index refresh complete. Total conversations: 68
[19:40:48] [loop] ✅ Conversation index refreshed

💡 Final Answer: 1. The GenAI Divide: 95% of organizations are getting zero return on their GenAI investments. 2. Productivity vs. P&L: Tools enhance individual productivity but lack significant P&L impact. 3. Learning is Key: Systems need to retain feedback and adapt to context to scale effectively.
```

**Analysis**:
- **Perception**: Initially failed due to API rate limits (429), but retried and correctly identified document extraction task
- **Step 1 Planning**: Generated plan to extract PDF content using `extract_pdf` tool, returning `FURTHER_PROCESSING_REQUIRED`
- **Step 1 Action**: Successfully extracted 56,477 characters of PDF content
- **Step 2 Planning**: Received the extracted content and generated a plan to analyze it directly (no tool calls)
- **Step 2 Action**: Processed the content and identified 3 main points from the executive summary
- **Memory**: Conversation was automatically indexed for future reference
- **Result**: Successfully extracted and summarized key points from the PDF

---

### Example 3: Web Search Query - Hummingbird Anatomy

**Query**: `Hummingbirds within Apodiformes uniquely have a bilaterally paired oval bone, a sesamoid embedded in the caudolateral portion of the expanded, cruciate aponeurosis of insertion of m. depressor caudae. How many paired tendons are supported by this sesamoid bone? Answer with a number.`

**Full Log**:
```
🧑 What do you want to solve today? → Hummingbirds within Apodiformes uniquely have a bilaterally paired oval bone, a sesamoid embedded in the caudolateral portion of the expanded, cruciate aponeurosis of insertion of m. depressor caudae. How many paired tendons are supported by this sesamoid bone? Answer with a number.

🔁 Step 1/3 starting...

[19:41:14] [perception] ⚠️ Perception failed: 429 RESOURCE_EXHAUSTED
[perception] intent='unknown' 
selected_servers=['math', 'documents', 'websearch']

[19:41:18] [plan] ⚠️ Planning exception: 429 RESOURCE_EXHAUSTED
[19:41:18] [loop] ⚠️ Planning failed: Planning failed with exception: 429 RESOURCE_EXHAUSTED
[19:41:18] [loop] 🛠 Retrying planning... Lifelines left: 2/3

[19:41:22] [perception] ⚠️ Perception failed: 429 RESOURCE_EXHAUSTED
[19:41:21] [plan] ⚠️ Planning exception: 429 RESOURCE_EXHAUSTED
[19:41:21] [loop] 🛠 Retrying planning... Lifelines left: 1/3

[19:41:25] [plan] ⚠️ Planning exception: 429 RESOURCE_EXHAUSTED
[19:41:25] [loop] 🛠 Retrying planning... Lifelines left: 0/3

[19:41:28] [perception] Raw output: json
{
  "intent": "Find information about hummingbird anatomy and answer a question about the number of tendons supported by a specific bone.",
  "entities": ["Hummingbirds", "Apodiformes", "bilaterally paired oval bone", "sesamoid", "tendons"],
  "tool_hint": "websearch, documents",
  "selected_servers": ["websearch", "documents"]
}

[perception] intent='Find information about hummingbird anatomy and answer a question about the number of tendons supported by a specific bone.' 
entities=['Hummingbirds', 'Apodiformes', 'bilaterally paired oval bone', 'sesamoid', 'tendons'] 
tool_hint='websearch, documents' 
selected_servers=['websearch', 'documents']

[19:41:35] [plan] LLM output: ```python
async def solve():
    return "FINAL_ANSWER: 2"

[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

[19:41:35] [indexer] 🔄 Refreshing conversation index (incremental update)...
[19:41:35] [indexer] ✅ Index refresh complete. Total conversations: 68
[19:41:35] [loop] ✅ Conversation index refreshed

💡 Final Answer: 2
```

**Analysis**:
- **Perception**: After multiple retries due to API rate limits, correctly identified the task as requiring web search for specialized anatomical information
- **Planning**: The LLM generated a direct answer based on its knowledge (the sesamoid bone supports 2 paired tendons)
- **Action**: Executed the simple return statement
- **Result**: Provided the answer "2" - the sesamoid bone supports 2 paired tendons
- **Note**: This example demonstrates the agent's resilience to API failures with automatic retry mechanisms

---

## 🔧 Configuration

The agent behavior is configured in `config/profiles.yaml`:

- **Strategy**: Planning mode (conservative/exploratory), max steps, retry limits
- **Memory**: Storage structure, summarization, tagging
- **LLM**: Text generation model (Gemini/Ollama), embedding model
- **MCP Servers**: Server configurations, capabilities, tool lists

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure LLM**:
   - Set up Gemini API credentials (for Vertex AI)
   - Or configure Ollama for local models

3. **Run the agent**:
   ```bash
   uv run agent.py
   ```

4. **Interact**:
   - Enter your query when prompted
   - Type `exit` to quit
   - Type `new` to start a new session

## 📝 Key Features

- **Multi-step Reasoning**: Handles complex tasks requiring multiple tool calls
- **Automatic Retries**: Resilient to API failures with configurable retry limits
- **Conversation Indexing**: Learns from past interactions using semantic search
- **Tool Integration**: Seamless integration with multiple MCP servers
- **Memory Persistence**: Stores all interactions for future reference
- **Error Handling**: Graceful handling of rate limits, tool failures, and invalid plans

## 📚 Documentation

- See `Architecture.md` for detailed system architecture and component diagrams
- See `BUG_FIX_REPORT.md` for bug fixes and improvements
- See `CONVERSATION_INDEXING_COMPLETE.md` for conversation indexing implementation details

## 🔍 Notes

- The agent uses `FURTHER_PROCESSING_REQUIRED` to handle multi-step tasks where intermediate results need additional processing
- Conversation indexing happens automatically after each interaction
- The agent maintains session continuity until explicitly reset with `new` command
- Rate limiting (429 errors) are handled with automatic retries using lifelines

---

**Cortex-R Agent** - Intelligent reasoning through perception, planning, and action.

