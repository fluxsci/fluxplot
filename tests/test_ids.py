from fluxplot import ids


def test_slugify_basic():
    assert ids.slugify("Control") == "control"
    assert ids.slugify("a__b  c") == "a-b-c"
    assert ids.slugify("  Dose-10  ") == "dose-10"
    assert ids.slugify("") == "series"
    assert ids.slugify("!!!") == "series"


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
