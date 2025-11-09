# modules/conversation_indexer.py

import json
import os
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
import time
from modules.model_manager import ModelManager

try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

@dataclass
class Conversation:
    """Represents a historical conversation."""
    session_id: str
    user_query: str
    final_answer: str
    timestamp: float
    tools_used: List[str]
    success: bool

class ConversationIndexer:
    """Indexes and searches historical conversations using semantic search."""
    
    def __init__(self, memory_dir: str = "memory", index_dir: str = "conversation_index"):
        self.memory_dir = Path(memory_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.index_file = self.index_dir / "conversations.index"
        self.metadata_file = self.index_dir / "conversations_metadata.json"
        self.index = None
        self.metadata: List[Dict] = []
        self.model_manager = ModelManager()
        
        # Load existing index if available
        self._load_index()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using configured model."""
        try:
            # Check if using Ollama for embeddings
            profile = self.model_manager.profile
            embed_model_key = profile.get("llm", {}).get("embedding", "nomic")
            models_config = self.model_manager.config
            
            if embed_model_key in models_config.get("models", {}):
                embed_config = models_config["models"][embed_model_key]
                
                if embed_config.get("type") == "ollama":
                    # Use Ollama embeddings
                    embed_url = embed_config.get("url", {}).get("embed", "http://localhost:11434/api/embeddings")
                    embed_model = embed_config.get("embedding_model", "nomic-embed-text")
                    
                    response = requests.post(
                        embed_url,
                        json={"model": embed_model, "prompt": text},
                        timeout=10
                    )
                    response.raise_for_status()
                    embedding = np.array(response.json()["embedding"], dtype=np.float32)
                    return embedding
                elif embed_config.get("type") == "gemini":
                    # Use Gemini embeddings
                    from google import genai
                    import os
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    api_key = os.getenv("GEMINI_API_KEY")
                    client = genai.Client(api_key=api_key)
                    
                    result = client.models.embed_content(
                        model=embed_config.get("embedding_model", "models/embedding-001"),
                        content=text
                    )
                    embedding = np.array(result.embedding, dtype=np.float32)
                    return embedding
            
            # Fallback to Ollama
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=10
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"], dtype=np.float32)
            
        except Exception as e:
            log("indexer", f"⚠️ Embedding error: {e}, using fallback")
            # Fallback: simple hash-based embedding (not ideal but works)
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to 128-dim vector (pad/truncate)
            # Repeat hash bytes to get enough data, then convert to float32 array
            repeated_bytes = (hash_bytes * 4)[:128 * 4]  # 128 floats = 512 bytes
            embedding = np.frombuffer(repeated_bytes, dtype=np.float32)[:128]
            return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                log("indexer", f"✅ Loaded index with {len(self.metadata)} conversations")
            else:
                log("indexer", "📝 No existing index found, will create new one")
        except Exception as e:
            log("indexer", f"⚠️ Error loading index: {e}, will create new one")
            self.index = None
            self.metadata = []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is not None and len(self.metadata) > 0:
                faiss.write_index(self.index, str(self.index_file))
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2)
                log("indexer", f"💾 Saved index with {len(self.metadata)} conversations")
        except Exception as e:
            log("indexer", f"⚠️ Error saving index: {e}")
    
    def _extract_conversations_from_session(self, session_file: Path) -> List[Conversation]:
        """Extract conversations from a session memory file."""
        conversations = []
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            
            # Find user query (from run_metadata or first item)
            user_query = None
            final_answer = None
            tools_used = []
            success = False
            timestamp = 0
            
            for item in items:
                if item.get("type") == "run_metadata":
                    # Try multiple ways to extract user query
                    user_query = (
                        item.get("user_query") or 
                        item.get("metadata", {}).get("user_query") or
                        (item.get("text", "").split("input:")[-1].split(" at")[0].strip() if "input:" in item.get("text", "") else None) or
                        item.get("text", "").split(":")[-1].strip()
                    )
                    timestamp = item.get("timestamp", 0)
                elif item.get("type") == "final_answer":
                    final_answer = item.get("final_answer") or item.get("text", "")
                    success = True
                elif item.get("type") == "tool_output" and item.get("tool_name"):
                    tools_used.append(item["tool_name"])
                    if item.get("success"):
                        success = True
            
            # Only index if we have both query and answer
            if user_query and final_answer and len(user_query) > 5 and len(final_answer) > 10:
                # Extract session ID from path or filename - handle multiple formats
                session_id = None
                
                # Try extracting from filename first
                if "session-" in session_file.name:
                    # Handle formats like:
                    # "session-2025-11-09-session-1762696001-228240.json"
                    # "session-1762696001-228240.json"
                    parts = session_file.stem.split("-")
                    if "session" in parts:
                        # Find the session part and get everything after it
                        session_idx = parts.index("session")
                        if session_idx + 1 < len(parts):
                            session_id = "-".join(parts[session_idx + 1:])
                    else:
                        session_id = session_file.stem.replace("session-", "")
                
                # Fallback: extract from path
                if not session_id:
                    parts = session_file.parts
                    for part in reversed(parts):
                        if "session-" in part:
                            # Extract the ID part after "session-"
                            session_id = part.replace("session-", "")
                            break
                
                # Last resort: use filename
                if not session_id:
                    session_id = session_file.stem.replace("session-", "")
                
                conversations.append(Conversation(
                    session_id=session_id or "unknown",
                    user_query=user_query[:500],  # Limit length
                    final_answer=final_answer[:2000],  # Limit length
                    timestamp=timestamp,
                    tools_used=list(set(tools_used)),  # Deduplicate
                    success=success
                ))
                log("indexer", f"✅ Extracted conversation: Q='{user_query[:50]}...' A='{final_answer[:50]}...'")
        except Exception as e:
            log("indexer", f"⚠️ Error reading {session_file}: {e}")
        
        return conversations
    
    def _scan_memory_files(self) -> List[Path]:
        """Scan memory directory for all session JSON files."""
        session_files = []
        if not self.memory_dir.exists():
            return session_files
        
        # Walk through memory/YYYY/MM/DD/session-*/session-*.json structure
        for year_dir in self.memory_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    for session_dir in day_dir.rglob("session-*.json"):
                        if session_dir.is_file():
                            session_files.append(session_dir)
        
        return session_files
    
    def index_all_conversations(self, force_rebuild: bool = False):
        """Index all conversations from memory files."""
        # Always scan for new files, even if index exists (for incremental updates)
        log("indexer", "🔍 Scanning memory files for conversations...")
        session_files = self._scan_memory_files()
        log("indexer", f"📁 Found {len(session_files)} session files")
        
        all_conversations = []
        indexed_session_ids = set()  # Track what we've already indexed
        
        # If index exists, track already indexed sessions
        if self.index is not None and self.metadata and not force_rebuild:
            indexed_session_ids = {m.get("session_id") for m in self.metadata if m.get("session_id")}
            log("indexer", f"📋 Already indexed {len(indexed_session_ids)} sessions")
        
        for session_file in session_files:
            # Extract session ID from path - handle different path formats
            session_id_from_path = None
            
            # Try to extract from filename first
            if "session-" in session_file.name:
                # Extract from filename like "session-2025-11-09-session-1762696001-228240.json"
                parts = session_file.stem.split("-")
                if "session" in parts:
                    # Find the last "session" occurrence and get everything after it
                    session_indices = [i for i, p in enumerate(parts) if p == "session"]
                    if session_indices:
                        last_session_idx = session_indices[-1]
                        if last_session_idx + 1 < len(parts):
                            session_id_from_path = "-".join(parts[last_session_idx + 1:])
                if not session_id_from_path:
                    session_id_from_path = session_file.stem.replace("session-", "")
            else:
                session_id_from_path = session_file.stem
            
            # Also try extracting from full path
            if not session_id_from_path:
                parts = session_file.parts
                for part in reversed(parts):
                    if "session-" in part:
                        session_id_from_path = part.replace("session-", "")
                        break
            
            # Check if already indexed (only for incremental updates)
            if not force_rebuild and session_id_from_path and session_id_from_path in indexed_session_ids:
                log("indexer", f"⏭️  Skipping already indexed session: {session_id_from_path}")
                continue
                
            conversations = self._extract_conversations_from_session(session_file)
            if conversations:
                log("indexer", f"📝 Extracted {len(conversations)} conversation(s) from {session_file.name}")
            all_conversations.extend(conversations)
            
            if force_rebuild:
                # Rebuild from scratch
                log("indexer", f"📝 Extracted {len(all_conversations)} conversations (rebuilding)")
                embeddings = []
                metadata = []
            else:
                # Incremental: add new conversations to existing index
                log("indexer", f"📝 Extracted {len(all_conversations)} new conversations")
                embeddings = []
                metadata = list(self.metadata) if self.metadata else []
            
            if not all_conversations and not metadata:
                log("indexer", "⚠️ No conversations found to index")
                return
            
            # Create embeddings for new conversations
            if all_conversations:
                log("indexer", "🔄 Creating embeddings...")
                for conv in all_conversations:
                    # Create searchable text: query + answer
                    searchable_text = f"Q: {conv.user_query}\nA: {conv.final_answer}"
                    embedding = self._get_embedding(searchable_text)
                    embeddings.append(embedding)
                    metadata.append({
                        "session_id": conv.session_id,
                        "user_query": conv.user_query,
                        "final_answer": conv.final_answer,
                        "timestamp": conv.timestamp,
                        "tools_used": conv.tools_used,
                        "success": conv.success
                    })
            
            # Build or update FAISS index
            if embeddings:
                if force_rebuild or self.index is None:
                    dim = len(embeddings[0])
                    self.index = faiss.IndexFlatL2(dim)
                    # Add all embeddings (existing + new)
                    if metadata and len(metadata) > len(embeddings):
                        # Need to re-embed existing ones too
                        log("indexer", "🔄 Re-embedding existing conversations...")
                        existing_embeddings = []
                        for m in metadata[:-len(all_conversations)]:
                            searchable_text = f"Q: {m['user_query']}\nA: {m['final_answer']}"
                            existing_embeddings.append(self._get_embedding(searchable_text))
                        embeddings = existing_embeddings + embeddings
                    self.index.add(np.stack(embeddings))
                else:
                    # Incremental: just add new embeddings
                    self.index.add(np.stack(embeddings))
                
                self.metadata = metadata
                self._save_index()
                log("indexer", f"✅ Indexed {len(metadata)} conversations")
            elif self.index is not None:
                log("indexer", f"✅ Using existing index with {len(self.metadata)} conversations")
        else:
            log("indexer", f"✅ Using existing index with {len(self.metadata)} conversations")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar conversations."""
        if self.index is None or len(self.metadata) == 0:
            # Try to load or build index
            self.index_all_conversations()
        
        if self.index is None or len(self.metadata) == 0:
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query).reshape(1, -1)
            
            # Search
            k = min(top_k, len(self.metadata))
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result["similarity_score"] = float(1 / (1 + dist))  # Convert distance to similarity
                    results.append(result)
            
            # Filter by minimum similarity threshold
            results = [r for r in results if r["similarity_score"] > 0.3]
            
            return results
        except Exception as e:
            log("indexer", f"⚠️ Search error: {e}")
            return []
    
    def get_relevant_context(self, query: str, top_k: int = 2) -> str:
        """Get formatted relevant historical context for LLM."""
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        context_parts = ["📚 Relevant Past Conversations:"]
        for i, result in enumerate(results, 1):
            # Truncate answer for context
            answer_preview = result['final_answer'][:300] + "..." if len(result['final_answer']) > 300 else result['final_answer']
            context_parts.append(
                f"{i}. Q: {result['user_query']}\n   A: {answer_preview}\n"
            )
        
        return "\n".join(context_parts)

# Global instance (lazy initialization)
_indexer_instance: Optional[ConversationIndexer] = None

def get_conversation_indexer() -> ConversationIndexer:
    """Get or create global conversation indexer instance."""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = ConversationIndexer()
        # Build index on first use (lazy, non-blocking)
        try:
            _indexer_instance.index_all_conversations()
        except Exception as e:
            log("indexer", f"⚠️ Error during initial indexing: {e}")
    return _indexer_instance

def refresh_conversation_index():
    """Manually refresh the conversation index (useful after new conversations)."""
    global _indexer_instance
    try:
        if _indexer_instance is not None:
            log("indexer", "🔄 Refreshing conversation index (incremental update)...")
            _indexer_instance.index_all_conversations(force_rebuild=False)  # Incremental update
            log("indexer", f"✅ Index refresh complete. Total conversations: {len(_indexer_instance.metadata)}")
        else:
            log("indexer", "🔄 Creating new conversation index...")
            get_conversation_indexer()  # Will index on creation
    except Exception as e:
        log("indexer", f"⚠️ Error refreshing index: {e}")
        import traceback
        traceback.print_exc()

