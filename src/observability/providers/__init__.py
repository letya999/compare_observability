"""Observability provider implementations."""

from .langsmith import LangSmithProvider
from .langfuse import LangfuseProvider
from .arize import ArizePhoenixProvider
from .opik import OpikProvider
from .braintrust import BraintrustProvider
from .laminar import LaminarProvider
from .agentops import AgentOpsProvider
from .evidently import EvidentlyProvider
from .evidently import EvidentlyProvider
from .logfire import LogfireProvider
from .weave import WeaveProvider
from .honeycomb import HoneycombProvider

# Registry of provider classes
PROVIDER_CLASSES = {
    "langsmith": LangSmithProvider,
    "langfuse": LangfuseProvider,
    "arize": ArizePhoenixProvider,
    "opik": OpikProvider,
    "braintrust": BraintrustProvider,
    "laminar": LaminarProvider,
    "agentops": AgentOpsProvider,
    "evidently": EvidentlyProvider,
    "evidently": EvidentlyProvider,
    "logfire": LogfireProvider,
    "weave": WeaveProvider,
    "honeycomb": HoneycombProvider,
}

__all__ = [
    "LangSmithProvider",
    "LangfuseProvider",
    "ArizePhoenixProvider",
    "OpikProvider",
    "BraintrustProvider",
    "LaminarProvider",
    "AgentOpsProvider",
    "EvidentlyProvider",
    "LogfireProvider",
    "WeaveProvider",
    "HoneycombProvider",
    "PROVIDER_CLASSES",
]
