"""Explicit teacher-forced diagnostic clone; never used by production generation."""

import ast
import inspect
import textwrap
import types


def clone_for_prefix(bound, prefix, decisions, source_path, *, guard_resolver=None):
    """After unchanged selection, feed prefix tokens into gen/cache/guard consistently.

    The original callable and logits remain untouched. The clone records actual
    selection independently of the fed prefix; it is not an OCR parity control.
    """
    original = inspect.unwrap(getattr(bound, "__func__", bound))
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    node = tree.body[0]
    loops = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step"
    ]
    if len(loops) != 1:
        raise ValueError("Unverified generation loop")
    loop = loops[0]
    positions = [
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "t_first is None"
    ]
    if len(positions) != 1:
        raise ValueError("Unverified token-selection boundary")
    inserted = ast.parse("""
_nr_decisions.append({'step': step, 'actual_selected': nxt,
                      'fed_token': _nr_prefix[step] if step < len(_nr_prefix) else nxt})
if step < len(_nr_prefix):
    nxt = _nr_prefix[step]
""").body
    loop.body[positions[0] : positions[0]] = inserted
    ast.fix_missing_locations(tree)
    source_path.write_text(
        "# Local diagnostic clone of immutable model function.\n" + ast.unparse(tree),
        encoding="utf-8",
    )
    namespace = dict(original.__globals__, _nr_prefix=list(prefix), _nr_decisions=decisions)
    if guard_resolver is not None:

        def current_guard(*args, **kwargs):
            return guard_resolver()(*args, **kwargs)

        namespace["_uocr_page_count_guard"] = current_guard
    exec(compile(source_path.read_text("utf-8"), str(source_path), "exec"), namespace)
    function = namespace[node.name]
    return types.MethodType(function, bound.__self__) if hasattr(bound, "__self__") else function
