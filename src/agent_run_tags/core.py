"""Tag and filter agent runs with arbitrary string labels.

:class:`RunTags` holds the mutable tag set for one run.
:class:`TagFilter` composes tag predicates (all-of, any-of, none-of).
:class:`RunRegistry` maintains a named collection of runs and their tags.

Example::

    from agent_run_tags import RunRegistry, TagFilter

    registry = RunRegistry()
    registry.register("run-1", tags={"model:gpt-4", "env:prod"})
    registry.register("run-2", tags={"model:claude", "env:dev"})
    registry.register("run-3", tags={"model:gpt-4", "env:dev"})

    prod_filter = TagFilter.all_of("env:prod")
    print(registry.find(prod_filter))
    # ['run-1']

    gpt4_filter = TagFilter.all_of("model:gpt-4")
    print(registry.find(gpt4_filter))
    # ['run-1', 'run-3']
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class RunTags:
    """Mutable tag bag for a single agent run.

    Tags are arbitrary non-empty strings.  Order is not preserved;
    iteration yields tags in sorted order.

    Example::

        tags = RunTags({"env:prod", "model:claude"})
        tags.add("retry:1")
        assert tags.has_all("env:prod", "model:claude")
        assert "retry:1" in tags
    """

    def __init__(self, tags: Iterable[str] | None = None) -> None:
        self._tags: set[str] = set()
        if tags:
            for t in tags:
                self._validate(t)
                self._tags.add(t)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, *tags: str) -> RunTags:
        """Add one or more tags.  Returns ``self`` for chaining."""
        for t in tags:
            self._validate(t)
            self._tags.add(t)
        return self

    def remove(self, *tags: str) -> RunTags:
        """Remove tags (silently ignores missing ones).  Returns ``self``."""
        for t in tags:
            self._tags.discard(t)
        return self

    def clear(self) -> RunTags:
        """Remove all tags.  Returns ``self``."""
        self._tags.clear()
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has(self, tag: str) -> bool:
        """Return ``True`` if *tag* is present."""
        return tag in self._tags

    def has_all(self, *tags: str) -> bool:
        """Return ``True`` if every supplied tag is present."""
        return all(t in self._tags for t in tags)

    def has_any(self, *tags: str) -> bool:
        """Return ``True`` if at least one supplied tag is present."""
        return any(t in self._tags for t in tags)

    def has_none(self, *tags: str) -> bool:
        """Return ``True`` if none of the supplied tags are present."""
        return not any(t in self._tags for t in tags)

    def all(self) -> frozenset[str]:
        """Return all tags as a :class:`frozenset`."""
        return frozenset(self._tags)

    def as_list(self) -> list[str]:
        """Return all tags as a sorted list."""
        return sorted(self._tags)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {"tags": self.as_list()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunTags:
        """Reconstruct :class:`RunTags` from a plain dict."""
        return cls(data.get("tags", []))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, tag: object) -> bool:
        return tag in self._tags

    def __len__(self) -> int:
        return len(self._tags)

    def __iter__(self):  # type: ignore[override]
        return iter(sorted(self._tags))

    def __repr__(self) -> str:
        return f"RunTags({self.as_list()!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(tag: str) -> None:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Tag must be a non-empty string, got {tag!r}.")


class TagFilter:
    """Composable predicate for matching :class:`RunTags`.

    Construct filters with the class-method factories, then call
    :meth:`matches` to test a :class:`RunTags` instance.

    Example::

        f = TagFilter.all_of("env:prod", "model:gpt-4")
        assert f.matches(RunTags({"env:prod", "model:gpt-4", "retry:0"}))
    """

    def __init__(self, predicate) -> None:  # type: ignore[type-arg]
        self._predicate = predicate

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def all_of(cls, *tags: str) -> TagFilter:
        """All supplied tags must be present."""
        return cls(lambda rt: rt.has_all(*tags))

    @classmethod
    def any_of(cls, *tags: str) -> TagFilter:
        """At least one of the supplied tags must be present."""
        return cls(lambda rt: rt.has_any(*tags))

    @classmethod
    def none_of(cls, *tags: str) -> TagFilter:
        """None of the supplied tags may be present."""
        return cls(lambda rt: rt.has_none(*tags))

    @classmethod
    def combine(cls, *filters: TagFilter, require_all: bool = True) -> TagFilter:
        """Combine multiple filters with AND (default) or OR logic.

        Args:
            filters:     Filters to combine.
            require_all: If ``True`` (default), all filters must match (AND).
                         If ``False``, at least one must match (OR).
        """
        if require_all:
            return cls(lambda rt: all(f.matches(rt) for f in filters))
        return cls(lambda rt: any(f.matches(rt) for f in filters))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def matches(self, tags: RunTags) -> bool:
        """Return ``True`` if *tags* satisfies this filter."""
        return bool(self._predicate(tags))

    def __repr__(self) -> str:
        return "TagFilter(...)"


class RunRegistry:
    """Named registry of agent runs and their associated :class:`RunTags`.

    Example::

        registry = RunRegistry()
        registry.register("run-abc", tags={"env:prod"})
        registry.get("run-abc").add("model:claude")
        ids = registry.find(TagFilter.all_of("env:prod"))
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunTags] = {}
        # Preserve insertion order for deterministic find() results
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        run_id: str,
        *,
        tags: Iterable[str] | None = None,
    ) -> RunTags:
        """Register a run and return its :class:`RunTags`.

        If *run_id* is already registered the existing :class:`RunTags`
        instance is returned unchanged (tags are not replaced).

        Args:
            run_id: Unique identifier for the run.
            tags:   Initial set of tags (optional).

        Returns:
            The :class:`RunTags` for this run.
        """
        if run_id not in self._runs:
            self._runs[run_id] = RunTags(tags)
            self._order.append(run_id)
        return self._runs[run_id]

    def unregister(self, run_id: str) -> None:
        """Remove a run.  Silently ignores unknown *run_id*."""
        if run_id in self._runs:
            del self._runs[run_id]
            self._order.remove(run_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> RunTags | None:
        """Return the :class:`RunTags` for *run_id*, or ``None``."""
        return self._runs.get(run_id)

    def find(self, tag_filter: TagFilter) -> list[str]:
        """Return run IDs whose tags match *tag_filter* (insertion order)."""
        return [rid for rid in self._order if tag_filter.matches(self._runs[rid])]

    def find_by_tags(self, *tags: str, require_all: bool = True) -> list[str]:
        """Return run IDs that have the given tags (insertion order).

        Args:
            tags:        Tags to look for.
            require_all: If ``True`` (default), all tags must match.
                         If ``False``, any single tag is enough.
        """
        f = TagFilter.all_of(*tags) if require_all else TagFilter.any_of(*tags)
        return self.find(f)

    def all_run_ids(self) -> list[str]:
        """Return all registered run IDs in insertion order."""
        return list(self._order)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the registry to a plain dict."""
        return {"runs": {rid: self._runs[rid].to_dict() for rid in self._order}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRegistry:
        """Reconstruct a :class:`RunRegistry` from a plain dict."""
        registry = cls()
        for run_id, tags_data in data.get("runs", {}).items():
            registry.register(run_id, tags=tags_data.get("tags", []))
        return registry

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._runs)

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._runs

    def __repr__(self) -> str:
        return f"RunRegistry(runs={len(self._runs)})"
