# tests/conftest.py

import pytest
import sys
import asyncio
import matplotlib

if sys.platform.startswith('win'):
    policy = asyncio.WindowsSelectorEventLoopPolicy()
else:
    policy = asyncio.DefaultEventLoopPolicy()
matplotlib.use('Agg')

@pytest.fixture(scope='session')
def event_loop_policy():
    return policy
