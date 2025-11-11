"""Prompt-level security guard executed before query planning."""
from __future__ import annotations

import os
import sys

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.processor.security import SecurityScanner


class PromptSecurityGuard:
    """Reusable guard that scans incoming prompts for risky patterns."""

    def __init__(
        self,
        scanner: SecurityScanner | None = None,
        security_notice: str | None = None,
    ) -> None:
        self._scanner = scanner or SecurityScanner()
        self._security_notice = security_notice or "Security policy triggered. Unable to process the prompt."

    def enforce(self, prompt: str) -> str | None:
        findings = self._scanner.scan(prompt)
        if findings:
            return self._security_notice
        return None
