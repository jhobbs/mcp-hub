# TODO

## LLM Collaboration Cleanup
- [ ] Remove OpenAI and Anthropic client classes from lib/llm_collaboration.py
- [ ] Remove multi-LLM collaboration classes (CollaborationSession, LLMOrchestrator) from lib/llm_collaboration.py
- [ ] Clean up LLMProvider enum to keep only GOOGLE in lib/llm_collaboration.py
- [ ] Remove OpenAI and Anthropic API key fields from config/settings.py
- [ ] Remove LLM orchestrator references from main.py
- [ ] Delete examples/demo_collaboration.py file entirely