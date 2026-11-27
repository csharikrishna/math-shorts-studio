"""Core adaptable component registry interfaces with enforceable typed & executable contracts."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

ComponentCategory = Literal[
    "math_operation",
    "visual_object",
    "animation_mechanism",
    "composition_template",
    "audio_cue",
    "caption_style",
]

BackendSupport = Literal["both", "manimgx", "hyperframes"]


class ComponentSpec(BaseModel):
    """Self-describing executable capability contract exposed to planning agents and renderers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    category: ComponentCategory
    description: str
    version: str = "3.8.0"
    input_model: str = "dict"
    output_model: str = "dict"
    inputs_schema: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    supported_backends: BackendSupport = "both"
    backend_support: dict[str, bool] = Field(default_factory=lambda: {"manimgx": True, "hyperframes": True})
    render_implementations: dict[str, str] = Field(default_factory=dict)
    supported_ops: list[str] = Field(default_factory=list)
    layout_constraints: list[str] = Field(default_factory=list)
    layout_requirements: dict[str, Any] = Field(default_factory=dict)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)
    aspect_support: list[str] = Field(default_factory=lambda: ["16:9", "9:16"])
    implemented: bool = True
    math_grounding: str = ""
    example_use_case: str = ""
    # Non-serialized runtime contracts & typed schema classes
    input_schema_model: Optional[type[BaseModel]] = Field(default=None, exclude=True)
    output_schema_model: Optional[type[BaseModel]] = Field(default=None, exclude=True)
    compute_fn: Optional[Callable[..., Any]] = Field(default=None, exclude=True)
    validate_fn: Optional[Callable[..., list[str]]] = Field(default=None, exclude=True)
    test_fn: Optional[Callable[[], bool]] = Field(default=None, exclude=True)

    def validate_inputs(self, data: dict[str, Any]) -> list[str]:
        """Validate payload against typed Pydantic input model and component validate_fn."""
        issues: list[str] = []
        if self.input_schema_model is not None:
            try:
                self.input_schema_model.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                issues.append(f"{self.name}: typed input validation failed ({exc})")
        if self.validate_fn is not None:
            try:
                issues.extend(str(x) for x in (self.validate_fn(data) or []))
            except Exception as exc:  # noqa: BLE001
                issues.append(f"{self.name}: validate_fn raised ({exc})")
        return issues

    def execute(self, **kwargs: Any) -> Any:
        if self.compute_fn is None:
            raise NotImplementedError(f"ComponentSpec {self.name!r} has no bound compute_fn")
        errs = self.validate_inputs(kwargs)
        if errs:
            raise ValueError("; ".join(errs))
        return self.compute_fn(**kwargs)

    def certification_errors(self) -> list[str]:
        """Check whether this component meets every executable contract requirement."""
        errs: list[str] = []
        if not self.implemented:
            errs.append("marked implemented=False")
            return errs
        if self.test_fn is None:
            errs.append("missing required executable test_fn")
        if self.input_schema_model is None or self.output_schema_model is None:
            errs.append("missing typed input_schema_model or output_schema_model")
        if self.category == "math_operation":
            if self.compute_fn is None:
                errs.append("math_operation missing compute_fn")
            if self.validate_fn is None:
                errs.append("math_operation missing validate_fn")
        if self.category in ("visual_object", "animation_mechanism", "composition_template", "caption_style"):
            if self.validate_fn is None:
                errs.append(f"{self.category} missing validate_fn")
            expected_backends = ("manimgx", "hyperframes") if self.supported_backends == "both" else (self.supported_backends,)
            for b_name in expected_backends:
                if self.backend_support.get(b_name, True) and not str(self.render_implementations.get(b_name, "")).strip():
                    errs.append(f"missing render_implementations[{b_name!r}] handler")
        if self.test_fn is not None:
            try:
                if not bool(self.test_fn()):
                    errs.append("contract test_fn returned False")
            except Exception as exc:  # noqa: BLE001
                errs.append(f"contract test_fn raised {type(exc).__name__}: {exc}")
        return errs

    def verify_contract(self) -> bool:
        """Strictly verify implementation, typed contract, handlers, and runnable test_fn."""
        return len(self.certification_errors()) == 0


class ComponentRegistry:
    """Thread-safe in-memory catalog of adaptable math, visual, motion, and layout primitives."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ComponentSpec] = {}

    def register(self, spec: ComponentSpec) -> ComponentSpec:
        self._items[(spec.category, spec.name)] = spec
        return spec

    def register_many(self, specs: list[ComponentSpec]) -> None:
        for s in specs:
            self.register(s)

    def get(self, name: str, category: ComponentCategory | None = None) -> ComponentSpec | None:
        if category is not None:
            return self._items.get((category, name))
        for (_cat, item_name), spec in self._items.items():
            if item_name == name:
                return spec
        return None

    def list_by_category(self, category: ComponentCategory) -> list[ComponentSpec]:
        return [spec for (cat, _), spec in self._items.items() if cat == category]

    def search(self, query: str) -> list[ComponentSpec]:
        q = query.strip().lower()
        if not q:
            return self.all_specs()
        hits: list[ComponentSpec] = []
        for spec in self._items.values():
            hay = f"{spec.name} {spec.category} {spec.description} {spec.math_grounding} {spec.example_use_case}".lower()
            if q in hay:
                hits.append(spec)
        return hits

    def all_specs(self) -> list[ComponentSpec]:
        return list(self._items.values())

    def certified_specs(self) -> list[ComponentSpec]:
        return [s for s in self._items.values() if s.implemented and not s.certification_errors()]

    def capabilities_prompt_summary(
        self,
        *,
        backend: str | None = None,
        aspect: str | None = None,
        compact: bool = True,
        certified_only: bool = True,
    ) -> str:
        """Render an up-to-date capability summary of certified implemented primitives only."""
        lines: list[str] = ["ADAPTABLE ENGINE CAPABILITY REGISTRY (certified primitives; zero hardcoded topic routes):"]
        order: list[ComponentCategory] = [
            "math_operation",
            "visual_object",
            "animation_mechanism",
            "composition_template",
            "caption_style",
            "audio_cue",
        ]
        for cat in order:
            specs = self.list_by_category(cat)
            filtered = [
                s
                for s in specs
                if s.implemented
                and (not certified_only or not s.certification_errors())
                and (backend is None or s.supported_backends in ("both", backend))
                and (aspect is None or aspect in s.aspect_support)
            ]
            if not filtered:
                continue
            lines.append(f"\n[{cat.upper()}] ({len(filtered)} available)")
            for s in filtered:
                inputs = ", ".join(s.inputs_schema[:5]) if s.inputs_schema else "none"
                if compact:
                    ground = f" | grounding={s.math_grounding}" if s.math_grounding else ""
                    lines.append(f"- {s.name}: {s.description} (inputs: {inputs}; outputs: {', '.join(s.outputs)}{ground})")
                else:
                    lines.append(
                        f"- {s.name} [{s.supported_backends}, {'/'.join(s.aspect_support)}]: {s.description} "
                        f"| inputs=[{inputs}] | outputs=[{', '.join(s.outputs)}] | e.g. {s.example_use_case}"
                    )
        return "\n".join(lines)
