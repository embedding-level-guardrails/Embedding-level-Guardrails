"""Jailbreak-template handling for RQ3 train-only augmentation.

HarmBench HumanJailbreaks templates mark the request slot with a positional
``str.format`` field ``{0}`` and escape literal braces as ``{{...}}``. HarmBench's
own baseline does not substitute: it emits ``f"{template}\\n\\n{behavior}"``,
which leaves ``{0}`` and doubled braces in the text. Here the request is
substituted into the slot instead.

Only one format is accepted: exactly one ``{0}`` field with no conversion or
format spec. Templates without a slot, with several slots or with other
fields are rejected explicitly, as is any output that still contains a
slot-like marker outside the substituted request.
"""

from __future__ import annotations

import ast
import hashlib
import string

SLOT = "0"
RESIDUAL_MARKERS = ("{0}", "[INSERT PROMPT HERE]", "[PROMPT HERE]", "<PROMPT>", "[PROMPT]")
_PROBE_REQUEST = "\x00probe-request\x00"


class TemplateFormatError(ValueError):
    pass


def check_template(template: str) -> None:
    try:
        fields = [(name, spec, conv) for _, name, spec, conv in string.Formatter().parse(template) if name is not None]
    except ValueError as e:  # unbalanced braces
        raise TemplateFormatError(f"unparseable template: {e}") from e
    if not fields:
        raise TemplateFormatError("no {0} slot")
    if len(fields) > 1:
        raise TemplateFormatError(f"{len(fields)} format fields, expected exactly one {{0}}")
    name, spec, conv = fields[0]
    if name != SLOT or spec or conv:
        raise TemplateFormatError(f"unsupported field {{{name}{'!' + conv if conv else ''}{':' + spec if spec else ''}}}")


def fill_template(template: str, request: str) -> str:
    """Substitute ``request`` into the template's single ``{0}`` slot."""
    check_template(template)
    filled = template.format(request)
    if request not in filled:
        raise TemplateFormatError("request not preserved verbatim")
    rest = filled.replace(request, "")
    residual = [m for m in RESIDUAL_MARKERS if m in rest]
    if residual:
        raise TemplateFormatError(f"residual placeholders {residual}")
    return filled


def template_id(source: str, index: int, template: str) -> str:
    digest = hashlib.sha1(template.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    return f"{source}:{index}:{digest}"


def usable_templates(templates: list[str], source: str = "harmbench_human_jailbreaks") -> tuple[list[tuple[str, str]], dict[str, str]]:
    """``([(template_id, template), ...], {template_id: rejection reason})``."""
    usable: list[tuple[str, str]] = []
    rejected: dict[str, str] = {}
    for index, template in enumerate(templates):
        tid = template_id(source, index, template)
        try:
            fill_template(template, _PROBE_REQUEST)
        except TemplateFormatError as e:
            rejected[tid] = str(e)
        else:
            usable.append((tid, template))
    return usable, rejected


def parse_harmbench_templates(source_text: str) -> list[str]:
    """Read the ``JAILBREAKS = [...]`` literal without executing remote code."""
    for node in ast.parse(source_text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JAILBREAKS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("JAILBREAKS list not found")
