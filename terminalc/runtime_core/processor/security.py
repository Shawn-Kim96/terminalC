import re
from typing import Iterable, Sequence


class SecurityScanner:
    """Detector for prompt injection attempts and secret leakage using regex + heuristics."""

    _DEFAULT_PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
        r"(?i)ignore (?:all )?previous instructions",
        r"(?i)disregard (?:the )?system prompt",
        r"(?i)override (?:the )?rules",
        r"(?i)forget(?:ting)? (?:all )?(?:previous|prior) (?:prompts?|instructions?)",
        r"(?i)reset (?:the )?(?:conversation|context)",
        r"(?i)reveal (?:the )?hidden prompt",
        r"(?i)leak (?:the )?system prompt",
        r"(?i)disable (?:the )?safety guardrails",
        r"(?i)jailbreak",
        r"(?i)perform .*?sql injection",
        r"(?i)perform .*?code injection",
    )
    _DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
        r"(?i)sk-[a-z0-9]{20,}",
        r"(?i)AKIA[0-9A-Z]{16}",
        r"(?i)AIza[0-9A-Za-z_\-]{35}",
        r"(?i)ghp_[0-9A-Za-z]{36}",
        r"(?i)eyJhbGciOi",
        r"(?i)-----BEGIN (?:RSA|DSA|EC) PRIVATE KEY-----",
        r"(?i)-----BEGIN PRIVATE KEY-----",
        r"(?i)xox[baprs]-[0-9A-Za-z]{10,48}",
        r"(?i)ssh-rsa [0-9A-Za-z/+]{100,}",
    )
    _KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
        (("forget", "previous", "prompt"), "prompt_injection:forget_previous_prompt"),
        (("forget", "all", "instructions"), "prompt_injection:forget_all_instructions"),
        (("return", "your", "weights"), "prompt_injection:return_weights"),
        (("dump", "model", "weights"), "prompt_injection:dump_model_weights"),
        (("share", "model", "parameters"), "prompt_injection:share_parameters"),
        (("expose", "training", "data"), "prompt_injection:expose_training_data"),
        (("show", "system", "prompt"), "prompt_injection:show_system_prompt"),
        (("reveal", "pipeline"), "prompt_injection:reveal_pipeline"),
        (("print", "pipeline"), "prompt_injection:print_pipeline"),
        (("describe", "your", "pipeline"), "prompt_injection:describe_pipeline"),
    )

    def __init__(
        self,
        prompt_injection_patterns: Sequence[str | re.Pattern[str]] | None = None,
        secret_leak_patterns: Sequence[str | re.Pattern[str]] | None = None,
    ) -> None:
        self._prompt_injection_patterns = tuple(self._compile_patterns(prompt_injection_patterns or self._DEFAULT_PROMPT_INJECTION_PATTERNS))
        self._secret_leak_patterns = tuple(self._compile_patterns(secret_leak_patterns or self._DEFAULT_SECRET_PATTERNS))

    @staticmethod
    def _compile_patterns(patterns: Iterable[str | re.Pattern[str]]) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            compiled.append(pattern if isinstance(pattern, re.Pattern) else re.compile(pattern))
        return compiled

    def scan(self, text: str) -> list[str]:
        findings: list[str] = []
        for regex in self._prompt_injection_patterns:
            if regex.search(text):
                findings.append(f"prompt_injection:{regex.pattern}")
        for regex in self._secret_leak_patterns:
            if regex.search(text):
                findings.append(f"secret_leak:{regex.pattern}")
        findings.extend(self._keyword_findings(text))
        return findings

    def _keyword_findings(self, text: str) -> list[str]:
        lowered = text.lower()
        results: list[str] = []
        for keywords, label in self._KEYWORD_RULES:
            if all(keyword in lowered for keyword in keywords):
                results.append(label)
        return results

