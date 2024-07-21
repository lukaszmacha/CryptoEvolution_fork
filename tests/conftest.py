# tests/conftest.py

import pytest
import sys
import asyncio

if sys.platform.startswith('win'):
    policy = asyncio.WindowsSelectorEventLoopPolicy()
else:
    policy = asyncio.DefaultEventLoopPolicy()

@pytest.fixture(scope='session')
def event_loop_policy():
    return policy
