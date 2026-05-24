# agent-run-tags

Tag and filter agent runs with arbitrary string labels.

Attach tags to named agent runs, then use composable filters to find runs by tag. Works as a lightweight labeling layer on top of any run tracking system.

## Install

```bash
pip install agent-run-tags
```

## Quick start

```python
from agent_run_tags import RunRegistry, TagFilter

registry = RunRegistry()
registry.register("run-1", tags={"model:gpt-4", "env:prod"})
registry.register("run-2", tags={"model:claude", "env:dev"})
registry.register("run-3", tags={"model:gpt-4", "env:dev"})

# Find all prod runs
prod = registry.find(TagFilter.all_of("env:prod"))
print(prod)  # ['run-1']

# Find all gpt-4 runs
gpt4 = registry.find(TagFilter.all_of("model:gpt-4"))
print(gpt4)  # ['run-1', 'run-3']

# Combine filters (AND)
f = TagFilter.combine(TagFilter.all_of("model:gpt-4"), TagFilter.none_of("env:prod"))
print(registry.find(f))  # ['run-3']
```

## API

### `RunTags`

Mutable tag bag for one run. Tags are arbitrary non-empty strings.

```python
tags = RunTags({"env:prod", "model:claude"})
tags.add("retry:1").add("batch:true")
tags.remove("retry:1")
tags.has_all("env:prod", "model:claude")  # True
tags.has_any("retry:1", "batch:true")     # True
tags.has_none("retry:1")                  # True
list(tags)       # sorted: ['batch:true', 'env:prod', 'model:claude']
tags.to_dict()   # {'tags': ['batch:true', 'env:prod', 'model:claude']}
```

### `TagFilter`

Composable predicate for `RunTags`.

| Factory | Description |
|---|---|
| `TagFilter.all_of(*tags)` | All tags must be present |
| `TagFilter.any_of(*tags)` | At least one tag must be present |
| `TagFilter.none_of(*tags)` | None of the tags may be present |
| `TagFilter.combine(*filters, require_all=True)` | AND (default) or OR of filters |

### `RunRegistry`

Named registry of runs and their tags.

| Method | Description |
|---|---|
| `register(run_id, *, tags)` | Register a run; returns its `RunTags` |
| `get(run_id)` | Get `RunTags` or `None` |
| `unregister(run_id)` | Remove a run (silent if missing) |
| `find(filter)` | Run IDs matching `TagFilter` |
| `find_by_tags(*tags, require_all=True)` | Shorthand tag search |
| `all_run_ids()` | All IDs in insertion order |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

## License

MIT
