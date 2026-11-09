"""The task protocol, so one harness can carry more than one constrained-generation task.

A task supplies everything that is specific to what the policy is asked to produce. The
harness supplies everything else: the group-relative update, the advantage, the matched-budget
guarantee, the counterfactual recomputation, and the analysis.

The contract a task must satisfy:

  PROMPTS          a fixed list of literal strings. It must NOT be generated at import or at
                   run time from a random source. The eval's identity is the number and order
                   of generate calls made after the eval seed is set, and the ledger's
                   matched-start check depends on it, so a task that drew its instances from
                   a random stream would shift the global RNG and void that check. Generate
                   instances once, offline, and paste them in.
  parse(text)      the completion as a structure, or None when nothing parses.
  verdict(x, lim)  True when x satisfies the constraint. Must accept None.
  residual(x, lim) how far outside the feasible set x lies, zero when feasible. Must be
                   monotone in that distance for the ordering claim to hold.
  task_term(x, lim) the rater reward, which is what pulls the policy across the constraint.
  synthesize(r, lim) an element whose residual is r. The unparseable rule prices a completion
                   that parsed to nothing at the worst residual present in its group, and
                   that needs a way back from a residual to something the penalties can score.
  UNPARSEABLE_TASK the task value a completion that parsed to nothing receives.
  ITEMS_KEY        the per-group key the trainer logs the parsed structures under.
  SCALAR_KEY       the per-step mean metric name.
  EVAL_SCALAR_KEY  the eval mean metric name.
  EVAL_ITEMS_KEY   the eval per-sample list name.

Nothing here changes the speed task's behaviour. `speed_task` re-exports the frozen functions
in `rewards.py`, whose definitions carry their own provenance citations and are not moved.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence


class TaskSpec:
    """A bundle of the task-specific callables. Deliberately a plain object, not a dataclass,
    so it stays importable without pulling anything else in."""

    def __init__(
        self,
        name: str,
        prompts: Sequence[str],
        parse: Callable[[str], Optional[Any]],
        verdict: Callable[[Any, float], bool],
        residual: Callable[[Any, float], float],
        task_term: Callable[[Any, float], float],
        synthesize: Callable[[float, float], Any],
        unparseable_task: float,
        items_key: str = "items",
        scalar_key: str = "mean_item",
        eval_scalar_key: str = "eval_mean_item",
        eval_items_key: str = "eval_items",
        limit_help: str = "the constraint limit",
    ):
        self.name = name
        self.PROMPTS = list(prompts)
        self.parse = parse
        self.verdict = verdict
        self.residual = residual
        self.task_term = task_term
        self.synthesize = synthesize
        self.UNPARSEABLE_TASK = unparseable_task
        self.ITEMS_KEY = items_key
        self.SCALAR_KEY = scalar_key
        self.EVAL_SCALAR_KEY = eval_scalar_key
        self.EVAL_ITEMS_KEY = eval_items_key
        self.LIMIT_HELP = limit_help

    def __repr__(self) -> str:  # pragma: no cover
        return f"TaskSpec({self.name!r}, {len(self.PROMPTS)} prompts)"

    def parser_for(self, prompt_index: int) -> Callable[[str], Optional[Any]]:
        """The parser for the prompt at this position. Tasks whose prompts each carry their own
        instance override this so the parsed structure knows which instance it belongs to."""
        return self.parse

    def summary_stat(self, parsed: Sequence[Any]) -> Optional[float]:
        """A single number describing a batch of parsed structures, for the logs only.
        Never read by the analysis. Overridden per task."""
        return None

    def instance_for(self, prompt_index: int) -> int:
        """Which instance a prompt poses. One for a task whose prompts share one output space."""
        return 0

    def output_table(self, limit: float, instance: int = 0) -> Dict[str, Any]:
        """Every parseable output of one instance, as arrays `task`, `residual` and `verdict`.

        The remedy arms need the largest reward a design can give and the best task value a
        feasible output reaches. Both come from this table by exhaustive enumeration, so neither
        is typed. Overridden per task."""
        raise NotImplementedError(f"task {self.name!r} does not enumerate its output space")

    def feasible_task_sup(self, limit: float, instance: int = 0) -> float:
        """The best task value any feasible output of this instance reaches."""
        table = self.output_table(limit, instance)
        feasible = table["task"][table["verdict"]]
        if feasible.size == 0:
            raise ValueError(f"instance {instance} of task {self.name!r} has no feasible output")
        return float(feasible.max())


_REGISTRY: Dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> TaskSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> TaskSpec:
    """Resolve a task by name. A pure dict read, so it consumes no randomness and can sit
    anywhere in the trainer without moving an RNG call."""
    if name not in _REGISTRY:
        _load_builtin()
    if name not in _REGISTRY:
        raise ValueError(f"unknown task {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> List[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def _load_builtin() -> None:
    import speed_task  # noqa: F401  (registers itself)
    try:
        import capacity_task  # noqa: F401
    except ImportError:  # pragma: no cover  (capacity task is optional)
        pass
    # ICML 2027 (November 2026). The bounded speed task of Block A' registers itself the same way.
    # Importing it consumes no randomness, so the speed and capacity paths are unchanged.
    try:
        import speed_bounded_task  # noqa: F401
    except ImportError:  # pragma: no cover  (the bounded speed task is optional)
        pass
