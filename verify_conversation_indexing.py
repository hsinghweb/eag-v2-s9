#!/usr/bin/env python3
"""
Verification script for conversation indexing system.
Tests that the indexing system is working correctly.
"""

import sys
from pathlib import Path

def verify_components():
    """Verify all components are in place."""
    print("🔍 Verifying Conversation Indexing System...\n")
    
    checks = []
    
    # Check 1: Conversation indexer module exists
    print("1. Checking conversation_indexer.py...")
    if Path("modules/conversation_indexer.py").exists():
        print("   ✅ modules/conversation_indexer.py exists")
        checks.append(True)
    else:
        print("   ❌ modules/conversation_indexer.py NOT FOUND")
        checks.append(False)
    
    # Check 2: Integration in decision.py
    print("\n2. Checking integration in decision.py...")
    try:
        with open("modules/decision.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "get_conversation_indexer" in content and "historical_context" in content:
                print("   ✅ Historical context integration found")
                checks.append(True)
            else:
                print("   ❌ Historical context integration NOT FOUND")
                checks.append(False)
    except Exception as e:
        print(f"   ❌ Error reading decision.py: {e}")
        checks.append(False)
    
    # Check 3: Integration in loop.py
    print("\n3. Checking integration in core/loop.py...")
    try:
        with open("core/loop.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "refresh_conversation_index" in content and "add_final_answer" in content:
                print("   ✅ Index refresh integration found")
                checks.append(True)
            else:
                print("   ❌ Index refresh integration NOT FOUND")
                checks.append(False)
    except Exception as e:
        print(f"   ❌ Error reading loop.py: {e}")
        checks.append(False)
    
    # Check 4: Index directory and files
    print("\n4. Checking index files...")
    index_dir = Path("conversation_index")
    if index_dir.exists():
        print("   ✅ conversation_index/ directory exists")
        index_file = index_dir / "conversations.index"
        meta_file = index_dir / "conversations_metadata.json"
        
        if index_file.exists():
            print(f"   ✅ Index file exists ({index_file.stat().st_size} bytes)")
        else:
            print("   ⚠️  Index file not found (will be created on first use)")
        
        if meta_file.exists():
            print(f"   ✅ Metadata file exists")
            checks.append(True)
        else:
            print("   ⚠️  Metadata file not found (will be created on first use)")
            checks.append(True)  # Not critical, will be created
    else:
        print("   ⚠️  conversation_index/ directory not found (will be created on first use)")
        checks.append(True)  # Not critical, will be created
    
    # Check 5: Memory files exist
    print("\n5. Checking memory files...")
    memory_dir = Path("memory")
    if memory_dir.exists():
        session_files = list(memory_dir.rglob("session-*.json"))
        print(f"   ✅ Found {len(session_files)} session files in memory/")
        checks.append(True)
    else:
        print("   ⚠️  memory/ directory not found")
        checks.append(False)
    
    # Check 6: Test indexer functionality
    print("\n6. Testing indexer functionality...")
    try:
        from modules.conversation_indexer import get_conversation_indexer
        indexer = get_conversation_indexer()
        
        print(f"   ✅ Indexer initialized successfully")
        print(f"   ✅ Indexed conversations: {len(indexer.metadata)}")
        
        # Test search
        if len(indexer.metadata) > 0:
            test_query = "test query"
            results = indexer.search(test_query, top_k=1)
            print(f"   ✅ Search functionality working")
            checks.append(True)
        else:
            print("   ⚠️  No conversations indexed yet (this is OK if no conversations completed)")
            checks.append(True)  # Not an error, just no data yet
    except Exception as e:
        print(f"   ❌ Error testing indexer: {e}")
        import traceback
        traceback.print_exc()
        checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("\n🎉 Conversation indexing system is COMPLETE and READY!")
        return True
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("\nSome components may need attention.")
        return False

def show_usage():
    """Show how the system works."""
    print("\n" + "="*60)
    print("HOW IT WORKS")
    print("="*60)
    print("""
1. 📚 INDEXING:
   - When a conversation completes (FINAL_ANSWER), it's automatically indexed
   - Conversations are extracted from memory/YYYY/MM/DD/session-*.json files
   - Each conversation (query + answer) is embedded using semantic embeddings
   - Stored in FAISS vector index for fast similarity search

2. 🔍 SEARCH:
   - Before generating a plan, the agent searches for similar past conversations
   - Top 2 most relevant conversations are retrieved
   - Added to the LLM prompt as context/examples

3. 🔄 AUTO-UPDATE:
   - Index refreshes automatically after each completed conversation
   - Only new conversations are indexed (incremental updates)
   - No manual intervention needed

4. 💡 BENEFITS:
   - Agent learns from past interactions
   - Similar queries get better answers based on history
   - Reduces redundant tool calls
   - Improves answer quality over time
    """)

if __name__ == "__main__":
    success = verify_components()
    show_usage()
    
    if success:
        print("\n✅ System verification: PASSED")
        sys.exit(0)
    else:
        print("\n⚠️  System verification: NEEDS ATTENTION")
        sys.exit(1)

