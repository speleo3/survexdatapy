import survexdata
import survexdata as m
from survexdata import Column
import csv
import io
import pathlib
import pytest
from xml.etree import ElementTree

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"


def test_findFile():
    path = DATA / "example.svx"
    assert m.findFile(DATA / "example.svx", case_sensitive=True) == path
    assert m.findFile(DATA / "example", case_sensitive=True) == path
    assert m.findFile(DATA / "Example.svx", case_sensitive=False) == path
    assert m.findFile(DATA / "Example", case_sensitive=False) == path
    assert m.findFile(DATA / "EXAMPLE.svx", case_sensitive=False) == path
    assert m.findFile(DATA / "EXAMPLE", case_sensitive=False) == path
    with pytest.raises(FileNotFoundError):
        m.findFile(DATA / "Example.svx", case_sensitive=True)
    with pytest.raises(FileNotFoundError):
        m.findFile(DATA / "Example", case_sensitive=True)
    with pytest.raises(FileNotFoundError):
        m.findFile(DATA / "EXAMPLE.svx", case_sensitive=True)
    with pytest.raises(FileNotFoundError):
        m.findFile(DATA / "EXAMPLE", case_sensitive=True)


def test_SpelixCsvFormatter():
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    parser.dump(survexdata.SpelixCsvFormatter(stream))

    stream.seek(0)
    rows = list(csv.reader(stream, delimiter=";"))

    assert len(rows) == 5
    assert rows[0] == [
        "Von", "Bis", "Länge", "Neigung", "Richtung", "L", "R", "O", "U", "RV", "Zeile"
    ]
    assert rows[1][:2] == ["A", "B"]
    assert rows[4][:2] == ["D", "E"]
    assert [float(v) for v in rows[1][2:5]] == [10.0, 30.0, 20.0]
    assert [float(v) for v in rows[4][2:5]] == [13.0, 39.0, 26.0]

def test_SvxFormatter():
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    parser.dump(survexdata.SvxFormatter(stream))

    stream.seek(0)
    rows = list(stream)

    assert len(rows) == 5
    assert rows[0].strip() == "*data default"

    cols = rows[4].split()

    assert cols[:2] == ["D", "E"]
    assert [float(v) for v in cols[2:5]] == [13.0, 26.0, 39.0]

def test_VisualTopoFormatter():
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    parser.dump(survexdata.VisualTopoFormatter(stream))

    stream.seek(0)
    rows = list(stream)

    for row in rows[11:]:
        assert len(row.rstrip()) == 93
        assert row[86] == 'N'
        assert row[38] == '.'


def test_VisualTopoFormatter_flags():
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "flags.svx")
    parser.dump(survexdata.VisualTopoFormatter(stream))

    rows = stream.getvalue().splitlines()

    assert rows[7] in ("Entree foo.a", "Entree foo.c")
    assert rows[11][88] == 'I'
    assert rows[12][88] == 'E'


def test_VisualTopoXmlFormatter():
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    parser.dump(survexdata.VisualTopoXmlFormatter(stream))

    root = ElementTree.fromstring(stream.getvalue())
    visees = root.findall("Mesures/Param/Visee")
    assert len(visees) == 5
    assert visees[0].get("Arr") == "A"
    assert visees[0].get("Dep") == "A"
    assert visees[1].get("Arr") == "B"
    assert visees[3].get("Dep") == "C"


def test_flags():
    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "flags.svx")
    assert parser._entrance_set == {"foo.a", "foo.c"}
    legs = list(parser.iterdata())
    assert legs[0].get(survexdata.Column.FLAGS) is None
    assert legs[1].get(survexdata.Column.FLAGS) == {"duplicate": True}


def test_case():
    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "case.svx")
    it = parser.iterdata()
    leg = next(it)
    assert leg[survexdata.Column.FROM] == "a"
    assert leg[survexdata.Column.TO] == "b"
    leg = next(it)
    assert leg[survexdata.Column.FROM] == "b"
    assert leg[survexdata.Column.TO] == "C"
    leg = next(it)
    assert leg[survexdata.Column.FROM] == "C"
    assert leg[survexdata.Column.TO] == "A"
    leg = next(it)
    assert leg[survexdata.Column.FROM] == "A"
    assert leg[survexdata.Column.TO] == "a"

def test_dataiter():
    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    assert sum(leg[survexdata.Column.TAPE]
               for leg in parser.iterdata()) == 46.0


def test_angleDiff():
    assert m.angleDiff(10, 15) == -5
    assert m.angleDiff(15, 10) == 5


def test_absAngleDiff():
    assert m.absAngleDiff(10, 15) == 5
    assert m.absAngleDiff(15, 10) == 5
    assert m.absAngleDiff(355, 5) == 10
    assert m.absAngleDiff(355, 5) == 10
    assert m.absAngleDiff(5, 355) == 10
    assert m.absAngleDiff(5, 355 + 360 * 2) == 10
    assert m.absAngleDiff(5, 355 + 360 * 3) == 10
    assert m.absAngleDiff(5, 355 + 360 * 4) == 10
    assert m.absAngleDiff(5, -3) == 8
    assert m.absAngleDiff(-5, 3) == 8


def test_tapeIsSame():
    assert m.tapeIsSame(0.42, 0.44)
    assert not m.tapeIsSame(0.42, 0.64)
    assert m.tapeIsSame(8.42, 8.44)
    assert not m.tapeIsSame(8.0, 9.5)


def test_shotIsSame():
    assert not m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3},
        {})
    assert m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3})
    assert m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.9, Column.COMPASS: 23.0, Column.CLINO: -12.0})
    assert m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.9, Column.COMPASS: 23.0, Column.CLINO: -12.0},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3})
    assert not m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 16.3, Column.COMPASS: 23.4, Column.CLINO: -12.3})
    assert not m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 26.4, Column.CLINO: -12.3})
    assert not m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -12.3},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 23.4, Column.CLINO: -16.3})
    assert m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 359.5, Column.CLINO: -0.5},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 0.5, Column.CLINO: 0.5})
    assert not m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 359.5, Column.CLINO: -0.5},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 3.5, Column.CLINO: 0.5})
    assert not m.shotIsSame(
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 359.5, Column.CLINO: -0.5},
        {Column.FROM: "a", Column.TO: "b", Column.TAPE: 12.3, Column.COMPASS: 0.5, Column.CLINO: 3.5})


REF_DUPS_ROWS = """
prev  : {FROM: 'A', TO: '..', TAPE: 10.1, COMPASS: 20.4, CLINO: 30.0, _FILE: '/dups.svx:3'}
  1/ 1: {FROM: 'A', TO: '..', TAPE: 10.11, COMPASS: 20.5, CLINO: 30.0, _FILE: '/dups.svx:4'}
  2/ 2: {FROM: 'A', TO: '..', TAPE: 10.12, COMPASS: 20.3, CLINO: 30.0, _FILE: '/dups.svx:5'}
 check: leg? /dups.svx:5
prev  : {FROM: 'A', TO: 'B', TAPE: 30.0, COMPASS: 40.0, CLINO: 50.0, _FILE: '/dups.svx:6'}
  D  1: {FROM: 'A', TO: 'B', TAPE: 30.0, COMPASS: 40.0, CLINO: 50.0, _FILE: '/dups.svx:7'}
  D  2: {FROM: 'A', TO: 'B', TAPE: 30.0, COMPASS: 40.0, CLINO: 50.1, _FILE: '/dups.svx:8'}
prev  : {FROM: 'B', TO: 'C', TAPE: 20.0, COMPASS: 30.0, CLINO: 40.6, _FILE: '/dups.svx:9'}
  L  1: {FROM: 'C', TO: 'D', TAPE: 20.0, COMPASS: 30.0, CLINO: 40.5, _FILE: '/dups.svx:10'}
 check: dup? /dups.svx:10
""".lstrip().splitlines()


@pytest.mark.parametrize("filename,rows_ref", [
    ("example.svx", []),
    ("dups-outer.svx", REF_DUPS_ROWS),
])
def test_FindDuplicateFormatter(filename: str, rows_ref: list):
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / filename)
    parser.dump(survexdata.FindDuplicateFormatter(stream))

    stream.seek(0)
    rows = [row.replace(DATA.as_posix(), "").rstrip() for row in stream]

    assert rows == rows_ref
