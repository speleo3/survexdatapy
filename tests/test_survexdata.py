import survexdata
import csv
import io
import pathlib
import pytest

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

def test_SpelixCsvFormatter():
    stream = io.StringIO()

    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    parser.dump(survexdata.SpelixCsvFormatter(stream))

    stream.seek(0)
    rows = list(csv.reader(stream, delimiter=";"))

    assert len(rows) == 5
    assert rows[0] == [
        "Von", "Bis", "Länge", "Neigung", "Richtung", "L", "R", "O", "U", "RV"
    ]
    assert rows[1][:2] == ["a", "b"]
    assert rows[4][:2] == ["d", "e"]
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

def test_dataiter():
    parser = survexdata.SvxParser()
    parser.parseFile(DATA / "example.svx")
    assert sum(leg[survexdata.Column.TAPE]
               for leg in parser.iterdata()) == 46.0
