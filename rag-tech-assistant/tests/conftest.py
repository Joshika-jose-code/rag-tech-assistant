import os
import sys

# ChatOpenAI raises at *construction* time (module import time, in our case,
# since app/graph/nodes.py builds the LLM client at module scope) if no API
# key is present. Tests never make real API calls, but the module still
# needs to import cleanly, so we set a dummy key before any app.* import.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-not-real")

import pytest
