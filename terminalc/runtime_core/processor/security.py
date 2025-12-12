import re
from typing import Iterable, Sequence


class SecurityScanner:
    """Detector for prompt injection attempts and secret leakage using regex + heuristics."""

    _DEFAULT_PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
        # Instruction override attempts
        r"(?i)ignore (?:all )?(?:previous )?(?:instructions?|rules?|prompts?|safety|filters?|guardrails?)",
        r"(?i)disregard (?:the )?(?:system prompt|rules|safety)",
        r"(?i)override (?:the )?(?:rules|safety|filters?)",
        r"(?i)forget(?:ting)? (?:all )?(?:previous|prior) (?:prompts?|instructions?)",
        r"(?i)reset (?:the )?(?:conversation|context)",
        r"(?i)disable (?:the )?(?:safety|filters?|guardrails?)",
        r"(?i)bypass (?:the )?(?:safety|security|filters?)",
        r"(?i)jailbreak",

        # System/prompt reveal attempts
        r"(?i)reveal (?:the )?(?:hidden |system )?(?:prompt|instructions?|rules)",
        r"(?i)leak (?:the )?system prompt",
        r"(?i)show (?:me )?(?:the |your )?(?:system |hidden )?prompt",
        r"(?i)print (?:the |your )?(?:system |full )?prompt",

        # Credential/secret extraction
        r"(?i)(?:reveal|show|expose|enumerate|list|display|return) .{0,30}(?:api[- ]?keys?|credentials?|secrets?|passwords?|tokens?)",
        r"(?i)(?:api[- ]?keys?|credentials?|secrets?|passwords?|tokens?).{0,30}(?:reveal|show|expose|enumerate|list|display|dump)",
        r"(?i)(?:cat|read|access|show) .{0,30}(?:\.env|credentials|secrets|config|\.aws)",

        # PII extraction
        r"(?i)(?:reveal|show|expose|list|return|dump) .{0,30}(?:email addresses?|user data|personal (?:info|data)|balances?|pii)",
        r"(?i)(?:email|user|personal|private).{0,30}(?:addresses?|data|information).{0,30}(?:paired|with|and)",

        # Database manipulation
        r"(?i)(?:drop|delete|truncate|alter) (?:table|database|schema)",
        r"(?i)(?:executable?|provide|show|write) (?:sql|query).{0,30}(?:drop|delete|truncate)",

        # Code execution attempts
        r"(?i)__import__\s*\(",
        r"(?i)eval\s*\(",
        r"(?i)exec\s*\(",
        r"(?i)os\.popen",
        r"(?i)subprocess\.",
        r"(?i)system\s*\(",
        r"(?i)execute .{0,30}(?:command|script|code|shell)",

        # Injection attacks
        r"(?i)perform .*?(?:sql|code|command) injection",
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
        # Prompt manipulation
        (("forget", "previous", "prompt"), "prompt_injection:forget_previous_prompt"),
        (("forget", "all", "instructions"), "prompt_injection:forget_all_instructions"),
        (("ignore", "safety", "filters"), "prompt_injection:ignore_safety_filters"),
        (("ignore", "all", "filters"), "prompt_injection:ignore_filters"),

        # Model/system exposure
        (("return", "your", "weights"), "prompt_injection:return_weights"),
        (("dump", "model", "weights"), "prompt_injection:dump_model_weights"),
        (("share", "model", "parameters"), "prompt_injection:share_parameters"),
        (("expose", "training", "data"), "prompt_injection:expose_training_data"),
        (("show", "system", "prompt"), "prompt_injection:show_system_prompt"),
        (("reveal", "pipeline"), "prompt_injection:reveal_pipeline"),
        (("print", "pipeline"), "prompt_injection:print_pipeline"),
        (("describe", "your", "pipeline"), "prompt_injection:describe_pipeline"),

        # Credential/secret extraction
        (("enumerate", "api", "key"), "credential_extraction:enumerate_api_keys"),
        (("enumerate", "credential"), "credential_extraction:enumerate_credentials"),
        (("reveal", "email", "addresses"), "pii_extraction:reveal_emails"),
        (("reveal", "user", "email"), "pii_extraction:reveal_user_emails"),
        (("show", "credentials"), "credential_extraction:show_credentials"),
        (("cat", "credentials"), "credential_extraction:cat_credentials"),
        (("cat", ".aws"), "credential_extraction:cat_aws_credentials"),

        # Database manipulation
        (("drop", "table"), "database_manipulation:drop_table"),
        (("delete", "table"), "database_manipulation:delete_table"),
        (("executable", "sql"), "database_manipulation:executable_sql"),
        (("provide", "sql", "drop"), "database_manipulation:provide_drop_sql"),

        # Code execution
        (("__import__", "os", "popen"), "code_execution:import_os_popen"),
        (("execute", "command"), "code_execution:execute_command"),
        (("run", "shell"), "code_execution:run_shell"),
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

