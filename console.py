#!/usr/bin/env python3
"""ClinicalMind Console — LangGraph Edition.

Usage:
    python console.py

First time:
    pip install langgraph langchain langchain-openai
    # Configure .env with your LLM API key
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from clinicalmind_lg.console import main

if __name__ == "__main__":
    asyncio.run(main())
