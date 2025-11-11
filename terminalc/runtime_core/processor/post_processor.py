"""LLM response validation, security scanning, and shaping."""
from __future__ import annotations

import os
import sys
from typing import Callable, Sequence

PROJECT_NAME = "terminalC"
PROJECT_DIR = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
sys.path.append(PROJECT_DIR)

from terminalc.runtime_core.models.runtime_models import LLMResult
from terminalc.runtime_core.processor.pre_processor import SecurityScanner


class LLMPostProcessor:
    def __init__(
        self,
        validator: Callable[[str], bool] | None = None,
        security_scanner: SecurityScanner | None = None,
        security_notice: str | None = None,
    ) -> None:
        self._validator = validator or (lambda text: len(text.strip()) > 0)
        self._security_scanner = security_scanner or SecurityScanner()
        self._security_notice = security_notice or "Security policy triggered. Unable to share the model response."

    def process(self, raw_text: str, model_name: str, token_usage: int | None = None) -> LLMResult:
        text = raw_text.strip()

        if not self._validator(text):
            return self._blocked_response(
                reason="validation",
                findings=[],
                model_name=model_name,
                token_usage=token_usage,
            )

        findings = self._security_scanner.scan(text)
        if findings:
            return self._blocked_response(
                reason="security",
                findings=findings,
                model_name=model_name,
                token_usage=token_usage,
            )

        return LLMResult(response_text=text, model_name=model_name, total_tokens=token_usage)

    def _blocked_response(
        self,
        reason: str,
        findings: Sequence[str],
        model_name: str,
        token_usage: int | None,
    ) -> LLMResult:
        message = ""
        if reason == "validation":
            message = "LLM result failed to pass validation"
        elif reason == "security":
            message = self._security_notice
        
        return LLMResult(response_text=message, model_name=model_name, total_tokens=token_usage)
