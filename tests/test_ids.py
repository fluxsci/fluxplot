from fluxplot import ids


def test_slugify_basic():
    assert ids.slugify("Control") == "control"
    assert ids.slugify("a__b  c") == "a-b-c"
    assert ids.slugify("  Dose-10  ") == "dose-10"
    assert ids.slugify("") == "series"
    assert ids.slugify("!!!").startswith("series-") and len(ids.slugify("!!!")) == len("series-") + 6


def test_slugify_transliterates_and_keeps_distinct_names_distinct():
    # Greek, micro, degree, percent, super/subscripts, accents
    assert ids.slugify("α") == "alpha" and ids.slugify("β") == "beta" and ids.slugify("Δ F/F") == "delta-f-f-" + ids.slugify("Δ F/F")[-6:]
    assert ids.slugify("µm") == "mum" and ids.slugify("37 °C") == "37-degc" and ids.slugify("% change") == "pct-change"
    assert ids.slugify("CO₂") == "co2" and ids.slugify("R²") == "r2" and ids.slugify("Café") == "cafe"
    assert ids.slugify("Mean ± SD") == "mean-pm-sd"
    # a name a soft mark punctuates keeps its plain slug; hard punctuation (brackets, slashes)
    # that the slug drops adds a hash, so two names differing only there stay distinct and stable
    assert ids.slugify("E. coli") == "e-coli" and ids.slugify("Yes, please") == "yes-please"
    a, b = ids.slugify("IL-6 (pg/mL)"), ids.slugify("IL-6 [pg/mL]")
    assert a != b and a.startswith("il-6-pg-ml-") and b.startswith("il-6-pg-ml-")
    assert ids.is_valid_segment(a) and ids.is_valid_segment(b)
    assert ids.slugify("IL-6 (pg/mL)") == a  # deterministic
    # the old rule, for idAliases
    assert ids.legacy_slugify("IL-6 (pg/mL)") == "il-6-pg-ml" and ids.legacy_slugify("α") == "series"


def test_series_id():
    assert ids.series_id("control") == "control"
    assert ids.series_id("control", "line") == "control.line"
    assert ids.series_id("control", "point", 3) == "control.point.3"


def test_structural_collision_disambiguates_series():
    # a series literally named "legend" must not shadow the structural root
    assert ids.series_id("legend") == "legend-series"
    assert ids.series_id("Axis", "line") == "axis-series.line"


def test_axis_id():
    assert ids.axis_id("x") == "axis.x"
    assert ids.axis_id("x", "title") == "axis.x.title"
    assert ids.axis_id("y", "ticklabel", 2) == "axis.y.ticklabel.2"


def test_join_validates_segments():
    import pytest

    with pytest.raises(ValueError):
        ids.join("bad.segment")  # dots only allowed as separators, not in a segment
    with pytest.raises(ValueError):
        ids.join("Upper")  # uppercase not allowed


def test_allocator_is_deterministic_and_collision_safe():
    a = ids.IdAllocator()
    assert a.take("control.line") == "control.line"
    assert a.take("control.line") == "control.line-2"
    assert a.take("control.line") == "control.line-3"
    assert a.take("legend") == "legend"
    assert a.take("legend") == "legend-2"
