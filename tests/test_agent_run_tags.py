"""Tests for agent-run-tags."""

from __future__ import annotations

import pytest

from agent_run_tags import RunRegistry, RunTags, TagFilter

# ---------------------------------------------------------------------------
# RunTags — construction
# ---------------------------------------------------------------------------


def test_run_tags_empty():
    tags = RunTags()
    assert len(tags) == 0
    assert tags.as_list() == []


def test_run_tags_from_set():
    tags = RunTags({"a", "b", "c"})
    assert len(tags) == 3
    assert "a" in tags


def test_run_tags_from_list():
    tags = RunTags(["x", "y"])
    assert len(tags) == 2


def test_run_tags_invalid_empty_string():
    with pytest.raises(ValueError):
        RunTags({""})


def test_run_tags_invalid_whitespace():
    with pytest.raises(ValueError):
        RunTags({"   "})


def test_run_tags_invalid_type():
    with pytest.raises((ValueError, AttributeError)):
        RunTags({123})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RunTags — add / remove / clear
# ---------------------------------------------------------------------------


def test_add_single():
    tags = RunTags()
    result = tags.add("env:prod")
    assert "env:prod" in tags
    assert result is tags  # chainable


def test_add_multiple():
    tags = RunTags()
    tags.add("a", "b", "c")
    assert len(tags) == 3


def test_add_duplicate_is_idempotent():
    tags = RunTags({"x"})
    tags.add("x")
    assert len(tags) == 1


def test_remove_existing():
    tags = RunTags({"a", "b"})
    result = tags.remove("a")
    assert "a" not in tags
    assert result is tags  # chainable


def test_remove_missing_is_silent():
    tags = RunTags({"a"})
    tags.remove("nope")  # should not raise
    assert len(tags) == 1


def test_clear():
    tags = RunTags({"a", "b"})
    result = tags.clear()
    assert len(tags) == 0
    assert result is tags  # chainable


# ---------------------------------------------------------------------------
# RunTags — has / has_all / has_any / has_none
# ---------------------------------------------------------------------------


def test_has_true():
    tags = RunTags({"env:prod"})
    assert tags.has("env:prod")


def test_has_false():
    tags = RunTags({"env:prod"})
    assert not tags.has("env:dev")


def test_has_all_true():
    tags = RunTags({"a", "b", "c"})
    assert tags.has_all("a", "b")


def test_has_all_false():
    tags = RunTags({"a", "b"})
    assert not tags.has_all("a", "b", "c")


def test_has_any_true():
    tags = RunTags({"a"})
    assert tags.has_any("a", "b")


def test_has_any_false():
    tags = RunTags({"c"})
    assert not tags.has_any("a", "b")


def test_has_none_true():
    tags = RunTags({"c"})
    assert tags.has_none("a", "b")


def test_has_none_false():
    tags = RunTags({"a"})
    assert not tags.has_none("a", "b")


# ---------------------------------------------------------------------------
# RunTags — all / as_list / iteration
# ---------------------------------------------------------------------------


def test_all_returns_frozenset():
    tags = RunTags({"x", "y"})
    result = tags.all()
    assert isinstance(result, frozenset)
    assert result == frozenset({"x", "y"})


def test_as_list_is_sorted():
    tags = RunTags({"c", "a", "b"})
    assert tags.as_list() == ["a", "b", "c"]


def test_iteration_is_sorted():
    tags = RunTags({"c", "a", "b"})
    assert list(tags) == ["a", "b", "c"]


def test_contains():
    tags = RunTags({"hello"})
    assert "hello" in tags
    assert "world" not in tags


# ---------------------------------------------------------------------------
# RunTags — serialisation
# ---------------------------------------------------------------------------


def test_to_dict():
    tags = RunTags({"env:prod", "model:gpt"})
    d = tags.to_dict()
    assert set(d["tags"]) == {"env:prod", "model:gpt"}


def test_from_dict_round_trip():
    tags = RunTags({"a", "b", "c"})
    restored = RunTags.from_dict(tags.to_dict())
    assert restored.all() == tags.all()


def test_from_dict_empty():
    tags = RunTags.from_dict({})
    assert len(tags) == 0


def test_repr():
    tags = RunTags({"b", "a"})
    r = repr(tags)
    assert "RunTags" in r
    assert "a" in r


# ---------------------------------------------------------------------------
# TagFilter — all_of / any_of / none_of
# ---------------------------------------------------------------------------


def test_all_of_match():
    tags = RunTags({"a", "b", "c"})
    assert TagFilter.all_of("a", "b").matches(tags)


def test_all_of_no_match():
    tags = RunTags({"a", "b"})
    assert not TagFilter.all_of("a", "b", "c").matches(tags)


def test_any_of_match():
    tags = RunTags({"a"})
    assert TagFilter.any_of("a", "b").matches(tags)


def test_any_of_no_match():
    tags = RunTags({"c"})
    assert not TagFilter.any_of("a", "b").matches(tags)


def test_none_of_match():
    tags = RunTags({"c"})
    assert TagFilter.none_of("a", "b").matches(tags)


def test_none_of_no_match():
    tags = RunTags({"a"})
    assert not TagFilter.none_of("a", "b").matches(tags)


# ---------------------------------------------------------------------------
# TagFilter — combine
# ---------------------------------------------------------------------------


def test_combine_all_and():
    tags = RunTags({"a", "b"})
    f = TagFilter.combine(
        TagFilter.all_of("a"),
        TagFilter.all_of("b"),
        require_all=True,
    )
    assert f.matches(tags)


def test_combine_all_and_fail():
    tags = RunTags({"a"})
    f = TagFilter.combine(
        TagFilter.all_of("a"),
        TagFilter.all_of("b"),
        require_all=True,
    )
    assert not f.matches(tags)


def test_combine_any_or():
    tags = RunTags({"a"})
    f = TagFilter.combine(
        TagFilter.all_of("a"),
        TagFilter.all_of("b"),
        require_all=False,
    )
    assert f.matches(tags)


def test_combine_any_or_fail():
    tags = RunTags({"c"})
    f = TagFilter.combine(
        TagFilter.all_of("a"),
        TagFilter.all_of("b"),
        require_all=False,
    )
    assert not f.matches(tags)


def test_tag_filter_repr():
    r = repr(TagFilter.all_of("a"))
    assert "TagFilter" in r


# ---------------------------------------------------------------------------
# RunRegistry — register / get / unregister
# ---------------------------------------------------------------------------


def test_register_returns_run_tags():
    registry = RunRegistry()
    rt = registry.register("run-1")
    assert isinstance(rt, RunTags)


def test_register_with_initial_tags():
    registry = RunRegistry()
    rt = registry.register("run-1", tags={"env:prod"})
    assert "env:prod" in rt


def test_register_duplicate_returns_existing():
    registry = RunRegistry()
    rt1 = registry.register("run-1", tags={"a"})
    rt2 = registry.register("run-1", tags={"b"})
    assert rt1 is rt2
    assert "a" in rt1
    assert "b" not in rt1


def test_registry_contains():
    registry = RunRegistry()
    registry.register("run-1")
    assert "run-1" in registry
    assert "run-2" not in registry


def test_get_existing():
    registry = RunRegistry()
    registry.register("run-1", tags={"x"})
    rt = registry.get("run-1")
    assert rt is not None
    assert "x" in rt


def test_get_missing():
    registry = RunRegistry()
    assert registry.get("nope") is None


def test_unregister():
    registry = RunRegistry()
    registry.register("run-1")
    registry.unregister("run-1")
    assert "run-1" not in registry
    assert len(registry) == 0


def test_unregister_missing_is_silent():
    registry = RunRegistry()
    registry.unregister("nope")  # no raise


def test_len():
    registry = RunRegistry()
    registry.register("r1")
    registry.register("r2")
    assert len(registry) == 2


def test_all_run_ids_insertion_order():
    registry = RunRegistry()
    registry.register("c")
    registry.register("a")
    registry.register("b")
    assert registry.all_run_ids() == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# RunRegistry — find / find_by_tags
# ---------------------------------------------------------------------------


def test_find_all_of():
    registry = RunRegistry()
    registry.register("r1", tags={"env:prod", "model:gpt"})
    registry.register("r2", tags={"env:dev", "model:gpt"})
    registry.register("r3", tags={"env:prod", "model:claude"})
    result = registry.find(TagFilter.all_of("env:prod"))
    assert result == ["r1", "r3"]


def test_find_any_of():
    registry = RunRegistry()
    registry.register("r1", tags={"a"})
    registry.register("r2", tags={"b"})
    registry.register("r3", tags={"c"})
    result = registry.find(TagFilter.any_of("a", "b"))
    assert result == ["r1", "r2"]


def test_find_none_of():
    registry = RunRegistry()
    registry.register("r1", tags={"a"})
    registry.register("r2", tags={"b"})
    result = registry.find(TagFilter.none_of("a"))
    assert result == ["r2"]


def test_find_empty_registry():
    registry = RunRegistry()
    assert registry.find(TagFilter.all_of("x")) == []


def test_find_by_tags_require_all():
    registry = RunRegistry()
    registry.register("r1", tags={"a", "b"})
    registry.register("r2", tags={"a"})
    result = registry.find_by_tags("a", "b", require_all=True)
    assert result == ["r1"]


def test_find_by_tags_require_any():
    registry = RunRegistry()
    registry.register("r1", tags={"a"})
    registry.register("r2", tags={"b"})
    result = registry.find_by_tags("a", "b", require_all=False)
    assert result == ["r1", "r2"]


# ---------------------------------------------------------------------------
# RunRegistry — serialisation
# ---------------------------------------------------------------------------


def test_registry_round_trip():
    registry = RunRegistry()
    registry.register("r1", tags={"env:prod", "model:gpt"})
    registry.register("r2", tags={"env:dev"})

    restored = RunRegistry.from_dict(registry.to_dict())
    assert len(restored) == 2
    assert "env:prod" in restored.get("r1")  # type: ignore[operator]
    assert "env:dev" in restored.get("r2")  # type: ignore[operator]


def test_registry_from_dict_empty():
    registry = RunRegistry.from_dict({})
    assert len(registry) == 0


def test_registry_repr():
    registry = RunRegistry()
    registry.register("r1")
    r = repr(registry)
    assert "RunRegistry" in r
    assert "1" in r
