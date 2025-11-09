# modules/loop.py

import asyncio
from modules.perception import run_perception
from modules.decision import generate_plan
from modules.action import run_python_sandbox
from modules.model_manager import ModelManager
from core.session import MultiMCP
from core.strategy import select_decision_prompt_path
from core.context import AgentContext
from modules.tools import summarize_tools
import re

try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

class AgentLoop:
    def __init__(self, context: AgentContext):
        self.context = context
        self.mcp = self.context.dispatcher
        self.model = ModelManager()

    async def run(self):
        max_steps = self.context.agent_profile.strategy.max_steps

        for step in range(max_steps):
            print(f"🔁 Step {step+1}/{max_steps} starting...")
            self.context.step = step
            lifelines_left = self.context.agent_profile.strategy.max_lifelines_per_step
            max_lifelines = lifelines_left  # Track original max for logging

            while lifelines_left >= 0:
                # === Perception ===
                user_input_override = getattr(self.context, "user_input_override", None)
                perception = await run_perception(context=self.context, user_input=user_input_override or self.context.user_input)

                print(f"[perception] {perception}")

                selected_servers = perception.selected_servers
                selected_tools = self.mcp.get_tools_from_servers(selected_servers)
                if not selected_tools:
                    log("loop", "⚠️ No tools selected — aborting step.")
                    if lifelines_left <= 0:
                        log("loop", f"⚠️ No lifelines left ({max_lifelines} attempts exhausted). Moving to next step or failing.")
                    break

                # === Planning ===
                tool_descriptions = summarize_tools(selected_tools)
                prompt_path = select_decision_prompt_path(
                    planning_mode=self.context.agent_profile.strategy.planning_mode,
                    exploration_mode=self.context.agent_profile.strategy.exploration_mode,
                )

                # Use user_input_override if available, otherwise use original user_input
                planning_input = user_input_override or self.context.user_input

                try:
                    plan = await generate_plan(
                        user_input=planning_input,
                        perception=perception,
                        memory_items=self.context.memory.get_session_items(),
                        tool_descriptions=tool_descriptions,
                        prompt_path=prompt_path,
                        step_num=step + 1,
                        max_steps=max_steps,
                    )
                    print(f"[plan] {plan}")
                except ValueError as e:
                    # Planning failed to generate valid solve() - retry with lifeline
                    log("loop", f"⚠️ Planning failed: {e}")
                    lifelines_left -= 1
                    if lifelines_left < 0:
                        log("loop", f"⚠️ All lifelines exhausted ({max_lifelines} attempts). Planning failed.")
                        break
                    log("loop", f"🛠 Retrying planning... Lifelines left: {lifelines_left}/{max_lifelines}")
                    continue
                except Exception as e:
                    # Other planning errors - retry
                    log("loop", f"⚠️ Planning error: {e}")
                    lifelines_left -= 1
                    if lifelines_left < 0:
                        log("loop", f"⚠️ All lifelines exhausted ({max_lifelines} attempts). Planning error.")
                        break
                    log("loop", f"🛠 Retrying planning after error... Lifelines left: {lifelines_left}/{max_lifelines}")
                    continue

                # === Execution ===
                if re.search(r"^\s*(async\s+)?def\s+solve\s*\(", plan, re.MULTILINE):
                    print("[loop] Detected solve() plan — running sandboxed...")

                    self.context.log_subtask(tool_name="solve_sandbox", status="pending")
                    result = await run_python_sandbox(plan, dispatcher=self.mcp)

                    success = False
                    if isinstance(result, str):
                        result = result.strip()
                        if result.startswith("FINAL_ANSWER:"):
                            success = True
                            self.context.final_answer = result
                            # Clear user_input_override since we got a final answer
                            if hasattr(self.context, "user_input_override"):
                                delattr(self.context, "user_input_override")
                            self.context.update_subtask_status("solve_sandbox", "success")
                            self.context.memory.add_tool_output(
                                tool_name="solve_sandbox",
                                tool_args={"plan": plan},
                                tool_result={"result": result},
                                success=True,
                                tags=["sandbox"],
                            )
                            # Save final answer to memory for indexing
                            final_answer_text = result.split("FINAL_ANSWER:")[1].strip() if "FINAL_ANSWER:" in result else result
                            self.context.memory.add_final_answer(final_answer_text)
                            # Ensure memory is saved before indexing
                            self.context.memory.save()
                            
                            # Trigger conversation indexing update (async, non-blocking)
                            try:
                                from modules.conversation_indexer import refresh_conversation_index
                                # Refresh index to include this new conversation (incremental update)
                                # Add a small delay to ensure file is written
                                import asyncio
                                await asyncio.sleep(0.1)  # Small delay to ensure file write completes
                                refresh_conversation_index()
                                log("loop", "✅ Conversation index refreshed")
                            except Exception as e:
                                log("loop", f"⚠️ Could not update conversation index: {e}")
                                import traceback
                                traceback.print_exc()
                            return {"status": "done", "result": self.context.final_answer}
                        elif result.startswith("FURTHER_PROCESSING_REQUIRED:"):
                            content = result.split("FURTHER_PROCESSING_REQUIRED:")[1].strip()
                            # Mark as success since FURTHER_PROCESSING_REQUIRED is expected behavior
                            success = True
                            self.context.update_subtask_status("solve_sandbox", "success")
                            self.context.memory.add_tool_output(
                                tool_name="solve_sandbox",
                                tool_args={"plan": plan},
                                tool_result={"result": result},
                                success=True,
                                tags=["sandbox", "further_processing"],
                            )
                            
                            # Check if we're on the last step - if so, return it to agent.py instead of continuing
                            if step >= max_steps - 1:
                                log("loop", f"⚠️ FURTHER_PROCESSING_REQUIRED on last step ({step+1}/{max_steps}). Returning to agent.py.")
                                return {"status": "done", "result": f"FURTHER_PROCESSING_REQUIRED: {content}"}
                            
                            # Format the override to make it clear content is provided and should be processed
                            # Truncate very long content to avoid token limits
                            content_preview = content[:4000] if len(content) > 4000 else content
                            if len(content) > 4000:
                                content_preview += f"\n\n[Content truncated - showing first 4000 characters of {len(content)} total]"
                            
                            self.context.user_input_override  = (
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
                            log("loop", f"📨 Forwarding intermediate result to next step:\n{self.context.user_input_override}\n\n")
                            log("loop", f"🔁 Continuing based on FURTHER_PROCESSING_REQUIRED — Step {step+1}/{max_steps} continues...")
                            break  # Step will continue to next step
                        elif result.startswith("[sandbox error:"):
                            success = False
                            error_msg = result.replace("[sandbox error:", "").strip().rstrip("]")
                            self.context.final_answer = f"FINAL_ANSWER: [Execution failed: {error_msg}]"
                            log("loop", f"⚠️ Sandbox error: {error_msg}")
                        else:
                            success = True
                            self.context.final_answer = f"FINAL_ANSWER: {result}"
                    else:
                        self.context.final_answer = f"FINAL_ANSWER: {result}"

                    if success:
                        self.context.update_subtask_status("solve_sandbox", "success")
                    else:
                        self.context.update_subtask_status("solve_sandbox", "failure")

                    self.context.memory.add_tool_output(
                        tool_name="solve_sandbox",
                        tool_args={"plan": plan},
                        tool_result={"result": result},
                        success=success,
                        tags=["sandbox"],
                    )

                    # Only retry if it's a failure, not if it's FURTHER_PROCESSING_REQUIRED (which already broke above)
                    if success and "FURTHER_PROCESSING_REQUIRED:" not in result:
                        return {"status": "done", "result": self.context.final_answer}
                    elif not success:
                        # Only retry on actual failures
                        lifelines_left -= 1
                        if lifelines_left < 0:
                            log("loop", f"⚠️ All lifelines exhausted ({max_lifelines} attempts). Step failed.")
                            # Break from lifeline loop, will continue to next step or fail at max_steps
                            break
                        log("loop", f"🛠 Retrying after failure... Lifelines left: {lifelines_left}/{max_lifelines}")
                        continue
                    else:
                        # This shouldn't happen, but if it does, break
                        log("loop", "⚠️ Unexpected state - breaking lifeline loop")
                        break
                else:
                    lifelines_left -= 1
                    if lifelines_left < 0:
                        log("loop", f"⚠️ All lifelines exhausted ({max_lifelines} attempts). Invalid plan step failed.")
                        # Break from lifeline loop, will continue to next step or fail at max_steps
                        break
                    log("loop", f"⚠️ Invalid plan detected — retrying... Lifelines left: {lifelines_left}/{max_lifelines}")
                    continue

        log("loop", "⚠️ Max steps reached without finding final answer.")
        # Check if we have pending FURTHER_PROCESSING_REQUIRED
        user_input_override = getattr(self.context, "user_input_override", None)
        if user_input_override:
            # Return the override as FURTHER_PROCESSING_REQUIRED for agent.py to handle
            # Extract the content from the override (it contains the tool result)
            # The override format is: "Original user task: ...\n\nYour last tool produced this result:\n\n{content}\n\n..."
            marker = "Your last tool produced this result:\n\n"
            if marker in user_input_override:
                # Extract content between marker and next \n\n
                start_idx = user_input_override.find(marker) + len(marker)
                end_idx = user_input_override.find("\n\n", start_idx)
                if end_idx != -1:
                    content = user_input_override[start_idx:end_idx].strip()
                else:
                    # No next \n\n, take rest of string
                    content = user_input_override[start_idx:].strip()
            else:
                # Fallback: use the whole override
                content = user_input_override
            return {"status": "done", "result": f"FURTHER_PROCESSING_REQUIRED: {content}"}
        else:
            self.context.final_answer = "FINAL_ANSWER: [Max steps reached]"
            return {"status": "done", "result": self.context.final_answer}
