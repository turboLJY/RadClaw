"""
Agent definitions for RadClaw.

Each agent is a configuration bundle (AgentConfig) that pairs a role-specific
system prompt with the appropriate tool registry.  The agentic loop in
src.core.loop drives execution — agents themselves are stateless configs.
"""