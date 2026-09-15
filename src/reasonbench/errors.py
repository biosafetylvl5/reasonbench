"""Exit codes and the error types the CLI knows how to render."""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar


class ExitCode(IntEnum):
    """Process exit codes. Infrastructure failures outrank policy ones."""

    OK = 0
    INTERNAL = 1
    USAGE = 2
    NOT_FOUND = 3
    AUTH = 10
    API = 11
    BUDGET = 12
    TIMEOUT = 13
    GATE = 20
    INTERRUPTED = 130


class ReasonBenchError(Exception):
    """Anything the CLI should report as a message rather than a traceback."""

    exit_code: ClassVar[ExitCode] = ExitCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details


class UsageError(ReasonBenchError):
    """Bad invocation, or a config the tool will not accept."""

    exit_code: ClassVar[ExitCode] = ExitCode.USAGE


class ConfigError(UsageError):
    """A configuration file is missing, malformed, or inconsistent."""


class MissingAPIKeyError(ReasonBenchError):
    """No API key could be resolved for an endpoint."""

    exit_code: ClassVar[ExitCode] = ExitCode.AUTH


class ManifestError(ConfigError):
    """A run directory's manifest is absent or unreadable."""


class RunDirError(ReasonBenchError):
    """The path given is not a run directory."""

    exit_code: ClassVar[ExitCode] = ExitCode.NOT_FOUND


class SampleNotFoundError(ReasonBenchError):
    """No stored sample matched."""

    exit_code: ClassVar[ExitCode] = ExitCode.NOT_FOUND


class ArtifactNotFoundError(ReasonBenchError):
    """No stored artifact matched."""

    exit_code: ClassVar[ExitCode] = ExitCode.NOT_FOUND


class AmbiguousSampleError(UsageError):
    """A sample id prefix matched more than one sample."""


class BudgetExceededError(ReasonBenchError):
    """Spending passed the configured cap; the run stopped."""

    exit_code: ClassVar[ExitCode] = ExitCode.BUDGET


class RunFailedError(ReasonBenchError):
    """The run produced no usable samples."""

    exit_code: ClassVar[ExitCode] = ExitCode.API


class DeadlineExceededError(ReasonBenchError):
    """The wall-clock deadline expired."""

    exit_code: ClassVar[ExitCode] = ExitCode.TIMEOUT


class GateFailedError(ReasonBenchError):
    """At least one gate assertion was breached."""

    exit_code: ClassVar[ExitCode] = ExitCode.GATE
