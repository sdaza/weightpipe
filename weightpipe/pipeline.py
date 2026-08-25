"""One-object entry point: sampling design, weighting steps, and estimation."""

from collections.abc import Sequence
from typing import Any, Self

import pandas as pd

from weightpipe.design import Design
from weightpipe.diagnostics.balance import BalanceReport
from weightpipe.diagnostics.balance import balance as _balance
from weightpipe.diagnostics.margins import margins as _margins
from weightpipe.estimate import estimate as _estimate
from weightpipe.recipe import Recipe
from weightpipe.result import WeightResult
from weightpipe.result import collect_weights as _collect_weights


class WeightPipe:
    """Survey weighting from design inputs to estimates.

    Pass the sampling inputs once (the design kind is inferred), optionally
    chain adjustment steps, then read weights or estimates. Weights are
    computed lazily and cached, so estimating without any step works:

    >>> pipe = WeightPipe(df, N=10_000)
    >>> pipe.estimate("y", variance="jackknife")  # doctest: +SKIP

    Steps return a new pipe, leaving the original untouched:

    >>> weighted = (
    ...     WeightPipe(df, weight="pw", psu="psu", strata="stratum")
    ...     .calibrate(method="raking", proportions=props)
    ...     .trim(max_ratio=5.0, reference="value")
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        weight: str | None = None,
        strata: str | None = None,
        psu: str | None = None,
        N: float | int | None = None,
        N_h: dict[Any, float] | pd.Series | None = None,
        probabilities: str | Sequence[str] | None = None,
        stage_weights: str | Sequence[str] | None = None,
        unit_id: str | None = None,
        copy: bool = True,
        fit_options: dict[str, Any] | None = None,
        _recipe: Recipe | None = None,
        _design: Design | None = None,
    ) -> None:
        if _recipe is not None and _design is not None:
            self._design = _design
            self._recipe = _recipe
        else:
            self._design = Design(
                data,
                weight=weight,
                strata=strata,
                psu=psu,
                N=N,
                N_h=N_h,
                probabilities=probabilities,
                stage_weights=stage_weights,
                copy=copy,
            )
            self._recipe = Recipe.from_design(self._design, unit_id=unit_id)
        self._fit_options: dict[str, Any] = dict(fit_options or {})
        self._result: WeightResult | None = None

    # -- introspection ----------------------------------------------------

    @property
    def design(self) -> Design:
        return self._design

    @property
    def recipe(self) -> Recipe:
        return self._recipe

    @property
    def kind(self) -> str:
        return self._design.kind

    @property
    def steps(self) -> list[str]:
        return [s.name for s in self._recipe.steps]

    def __repr__(self) -> str:
        steps = " -> ".join(self.steps) if self.steps else "no steps"
        return f"WeightPipe(kind={self.kind!r}, n={len(self._recipe.data)}, {steps})"

    # -- chaining ---------------------------------------------------------

    def _with_recipe(self, recipe: Recipe) -> Self:
        return type(self)(
            recipe.data,
            _recipe=recipe,
            _design=self._design,
            fit_options=self._fit_options,
        )

    def options(self, **fit_options: Any) -> Self:
        """Set ``prep`` options (``min_cell_n``, ``max_factor``, ``warn``)."""
        out = self._with_recipe(self._recipe)
        out._fit_options = {**self._fit_options, **fit_options}
        return out

    def unknown_eligibility(self, **kwargs: Any) -> Self:
        return self._with_recipe(self._recipe.step_unknown_eligibility(**kwargs))

    def drop_ineligible(self, **kwargs: Any) -> Self:
        return self._with_recipe(self._recipe.step_drop_ineligible(**kwargs))

    def nonresponse(self, **kwargs: Any) -> Self:
        return self._with_recipe(self._recipe.step_nonresponse(**kwargs))

    def calibrate(self, **kwargs: Any) -> Self:
        return self._with_recipe(self._recipe.step_calibrate(**kwargs))

    def trim(self, **kwargs: Any) -> Self:
        return self._with_recipe(self._recipe.step_trim(**kwargs))

    def trim_weights(self, **kwargs: Any) -> Self:
        return self._with_recipe(self._recipe.step_trim_weights(**kwargs))

    # -- running ----------------------------------------------------------

    def fit(self, **kwargs: Any) -> WeightResult:
        """Run the steps and cache the result (alias of ``prep``)."""
        options = {**self._fit_options, **kwargs}
        self._result = self._recipe.prep(**options)
        return self._result

    prep = fit

    @property
    def result(self) -> WeightResult:
        """Fitted result, computing it on first access."""
        if self._result is None:
            self.fit()
        assert self._result is not None
        return self._result

    @property
    def weights(self) -> pd.Series:
        return self.result.weights

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.result.diagnostics

    @property
    def alerts(self) -> tuple[str, ...]:
        return self.result.alerts

    def collect_weights(self, **kwargs: Any) -> pd.DataFrame:
        """Weights (and optional intermediates) as a tidy frame."""
        return _collect_weights(self.result, **kwargs)

    def margins(
        self,
        variables: str | Sequence[str] | None = None,
        *,
        margins: dict[str, dict[str, float]] | None = None,
        proportions: dict[str, dict[str, float]] | None = None,
        population_size: float | None = None,
        targets: str | None = None,
        force1: bool = True,
    ) -> pd.DataFrame:
        """Current weighted category margins, optionally vs targets.

        Call anytime after fit. Pass ``targets='calibrate'`` to reuse the last
        calibrate step's targets, or pass ``margins=`` / ``proportions=``.
        """
        return _margins(
            self.result,
            variables,
            margins=margins,
            proportions=proportions,
            population_size=population_size,
            targets=targets,  # type: ignore[arg-type]
            force1=force1,
        )

    def balance(
        self,
        covariates: str | Sequence[str],
        *,
        target: pd.DataFrame | None = None,
        target_weight: str | None = None,
        means: dict[str, float] | None = None,
        proportions: dict[str, dict[str, float]] | None = None,
        sds: dict[str, float] | None = None,
        before: pd.Series | str | None = "base",
        threshold: float = 0.1,
    ) -> BalanceReport:
        """Covariate balance before vs after weighting (SMD / ASMD).

        Pass population microdata via ``target=``, or explicit ``means`` /
        ``proportions``. Compare base weights to final weights by default.
        """
        return _balance(
            self.result,
            covariates,
            target=target,
            target_weight=target_weight,
            means=means,
            proportions=proportions,
            sds=sds,
            before=before,  # type: ignore[arg-type]
            threshold=threshold,
        )

    def estimate(self, variable: str, **kwargs: Any) -> pd.DataFrame:
        """Estimate with recipe-aware bootstrap/jackknife, or linearized SE."""
        kwargs.setdefault("fitted", self.result)
        return _estimate(self._recipe, variable, **kwargs)
