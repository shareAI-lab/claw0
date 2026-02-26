# Agents

## Default Agent

The default agent handles all messages unless routing bindings direct
traffic to a specific agent.

## Multi-Agent Setup

In production OpenClaw, multiple agents can run simultaneously:
- Each agent has its own workspace, sessions, and memory
- Routing bindings determine which agent handles each message
- Agents are isolated: they cannot read each other's sessions

## Agent Communication

Agents do not communicate directly with each other.
Coordination happens through:
1. Shared workspace files (if configured)
2. The routing layer (bindings can be updated at runtime)
3. The human operator (via gateway API or CLI)
