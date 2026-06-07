"""Anthropic API call wrapper with retry on rate-limit errors."""

import time

import anthropic

RETRY_DELAYS = [15, 30, 60]   # seconds between retries on 429 / 529


def create_with_retry(client: anthropic.Anthropic, **kwargs):
    """Call client.messages.create with exponential backoff on rate-limit errors."""
    for attempt, delay in enumerate([0] + RETRY_DELAYS, 1):
        if delay:
            print(f"    rate limit — waiting {delay}s (attempt {attempt}/{len(RETRY_DELAYS)+1})")
            time.sleep(delay)
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt <= len(RETRY_DELAYS):
                continue
            raise
