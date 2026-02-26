# Tools Guide

## Available Tools

Your tools depend on which section you are running. Common tools include:

### File Tools
- **read_file**: Read file contents within the workspace
- **write_file**: Write content to a file (creates parent directories)
- **edit_file**: Replace exact text in a file (old_string must be unique)
- **list_directory**: List files and directories

### Memory Tools
- **memory_write**: Store important information for later recall
- **memory_search**: Search through stored memories by keyword

### System Tools
- **bash**: Execute shell commands (with safety checks)
- **get_current_time**: Get current date and time

## Usage Guidelines

- Always read a file before editing it
- Use memory_write proactively when users share preferences, facts, or decisions
- Use memory_search before answering questions about prior conversations
- Keep tool outputs concise -- the model context has limits
