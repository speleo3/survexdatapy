"""
Library for reading Survex data files (*.svx)

Copyright (C) Thomas Holder
License: BSD-2-Clause

Example:

    >>> parser = SvxParser()
    >>> parser.parseFile("test.svx")
    >>> parser.dump(SpelixCsvFormatter())

"""

import math
import re
import sys
import shlex
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as etree

NoSuchEnumError = KeyError


class UpperCaseStringEnum(Enum):
    """
    Base class for enums with upper case keys and (key == value).
    """
    def _generate_next_value_(name, start, count, last_values):
        assert name == name.upper()
        return name

    @classmethod
    def from_string(cls, value: str):
        return cls[value.upper()]


class Command(UpperCaseStringEnum):
    CASE = auto()
    DATA = auto()
    BEGIN = auto()
    END = auto()
    FLAGS = auto()
    INCLUDE = auto()
    EQUATE = auto()
    TITLE = auto()
    CS = auto()
    FIX = auto()
    ENTRANCE = auto()
    CALIBRATE = auto()
    UNITS = auto()
    EXPORT = auto()
    DATE = auto()
    INFER = auto()
    TEAM = auto()
    INSTRUMENT = auto()
    REF = auto()
    SD = auto()
    SET = auto()
    SOLVE = auto()
    ALIAS = auto()
    REQUIRE = auto()
    DECLINATION = auto()


class DataStyle(UpperCaseStringEnum):
    DEFAULT = auto()
    NORMAL = auto()
    DIVING = auto()
    CARTESIAN = auto()
    TOPOFIL = auto()
    CYLPOLAR = auto()
    NOSURVEY = auto()
    PASSAGE = auto()


class Column(UpperCaseStringEnum):
    FROM = auto()
    TO = auto()
    TAPE = auto()
    COMPASS = auto()
    COUNTER = auto()
    CLINO = auto()
    BACKTAPE = auto()
    BACKCOMPASS = auto()
    BACKCLINO = auto()
    FROMDEPTH = auto()
    TODEPTH = auto()
    DEPTHCHANGE = auto()
    IGNORE = auto()
    IGNOREALL = auto()
    STATION = auto()
    DIRECTION = auto()
    DEPTH = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    DX = auto()
    DY = auto()
    DZ = auto()
    DESCRIPTION = auto()
    DECLINATION = auto()
    X = auto()
    Y = auto()
    Z = auto()
    DATE = auto()  # custom
    NEWLINE = auto()
    FLAGS = auto()

    # aliases
    ALTITUDE = DZ
    BACKLENGTH = BACKTAPE
    BACKBEARING = BACKCOMPASS
    BACKGRADIENT = BACKCLINO
    BEARING = COMPASS
    CEILING = UP
    COUNT = COUNTER
    EASTING = DX
    FLOOR = DOWN
    GRADIENT = CLINO
    LENGTH = TAPE
    NORTHING = DY

    # DEBUG
    _FILE = auto()

    def __repr__(self):
        return self.value


class Units(UpperCaseStringEnum):
    YARDS = auto()
    FEET = auto()
    METRES = auto()
    DEGREES = auto()
    GRADS = auto()
    MINUTES = auto()
    PERCENT = auto()
    QUADS = auto()

    # aliases
    METRIC = METRES
    METERS = METRES
    DEGS = DEGREES
    PERCENTAGE = PERCENT
    QUADRANTS = QUADS


STATION_COLUMNS = [
    Column.FROM,
    Column.TO,
    Column.STATION,
]

DEFAULT_COLUMN_ORDER = [
    Column.FROM,
    Column.TO,
    Column.TAPE,
    Column.COMPASS,
    Column.CLINO,
]

DEFAULT_CALIBRATE: Dict[Column, Tuple[float, Units, float]] = {
    Column.TAPE: (0, Units.METRES, 1),
    Column.COMPASS: (0, Units.DEGREES, 1),
    Column.CLINO: (0, Units.DEGREES, 1),
    Column.DECLINATION: (0, Units.DEGREES, 1),
    Column.DEPTH: (0, Units.METRES, 1),
    Column.DEPTHCHANGE: (0, Units.METRES, 1),
}

DEFAULT_UNITS: Dict[Column, Tuple[float, Units]] = {
    Column.TAPE: (1, Units.METRES),
    Column.COMPASS: (1, Units.DEGREES),
    Column.CLINO: (1, Units.DEGREES),
    Column.DECLINATION: (1, Units.DEGREES),
    Column.DEPTH: (1, Units.METRES),
    Column.DEPTHCHANGE: (1, Units.METRES),
}

UNIT_CONVERT = {
    Units.METRES: 1.0,
    Units.FEET: 0.3048,
    Units.DEGREES: 1.0,
    Units.GRADS: 0.9,
}

NAMED_VALUES = {
    'DOWN': -90,
    'UP': 90,
    '-': 0,
    'LEVEL': 0,
}

ANONYMOUS_STATIONS = {'.', '..', '...'}

VT_MAXLEN = 11

DataDict = Dict[Column, Any]


def convert_unit(reading: float, unit: Units) -> float:
    """
    Convert reading in given unit to default unit.
    """
    if unit == Units.PERCENT:
        return math.degrees(math.atan(reading / 100))

    return reading * UNIT_CONVERT[unit]


def warn(*args, level="Warning"):
    """
    Print a warning to STDERR
    """
    print(f"{level}:", *args, file=sys.stderr)


def glob_case_insensitive(parent: Path, pattern: str) -> Iterable[Path]:
    """
    Python 3.12: parent.glob(pattern, case_sensitive=False)
    """
    pattern = re.sub(r'[a-zA-Z]',
                     lambda c: f'[{c.group(0).lower()}{c.group(0).upper()}]', pattern)
    return parent.glob(pattern)


def findFile(path: Path, *, case_sensitive: bool = True, suffix: str = ".svx") -> Path:
    """
    Find file, either as exact match, with added suffix, case insensitive or
    case insensitive with added suffix.
    """
    if path.exists():
        return path

    withsuffix = path.parent / (path.name + suffix)
    if withsuffix.exists():
        return withsuffix

    if not case_sensitive:
        for globbed in glob_case_insensitive(path.parent, path.name):
            return globbed

        for globbed in glob_case_insensitive(path.parent, path.name + suffix):
            return globbed

    raise FileNotFoundError(path)


class Formatter:
    """
    Base class for dumping Survex data.
    """
    def __init__(self, file=None):
        self.file = sys.stdout if file is None else file
        self._survey_mapping = {}
        self._name_mapping = {}

    def normalizeStationName(self, name: str) -> str:
        survey, dot, station = name.rpartition('.')
        if not survey:
            return name

        survey = self._survey_mapping.setdefault(survey,
                                                 len(self._survey_mapping))

        return f"{survey}{dot}{station}"

    def skipAnonymousStations(self) -> bool:
        return False

    def setEntrances(self, entrances: Iterable[str]):
        self.entrances = list(entrances)

    def printHeader(self):
        pass

    def printFooter(self):
        pass

    def printDate(self, data: dict):
        pass

    def printDeclination(self, data: dict):
        pass

    def printDataLine(self, data: dict):
        print(data[Column.FROM],
              data[Column.TO],
              data[Column.TAPE],
              data[Column.COMPASS],
              data[Column.CLINO],
              file=self.file)


class SvxFormatter(Formatter):
    def printHeader(self):
        print('*data default', file=self.file)


class SpelixCsvFormatter(Formatter):
    def skipAnonymousStations(self):
        return True

    def printHeader(self):
        print('"Von";"Bis";"Länge";"Neigung";"Richtung";"L";"R";"O";"U";"RV";"Zeile"',
              file=self.file)

    def printDataLine(self, data):
        print(
            ';'.join(f'"{data[col]}"'
                     for col in [Column.FROM, Column.TO]) + ';' +
            ';'.join(f'"{data[col]}"'
                     for col in [Column.TAPE, Column.CLINO, Column.COMPASS]) +
            ';"0"' * 5 + ';"' + data.get(Column._FILE, "") + '"',
            file=self.file)


def angleDiff(a: float, b: float) -> float:
    """
    Angle difference in the shorter direction.

    Args:
      a: Angle in degrees
      b: Angle in degrees
    """
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def absAngleDiff(a: float, b: float) -> float:
    """
    Absolute angle difference in the shorter direction.

    Args:
      a: Angle in degrees
      b: Angle in degrees
    """
    return abs(angleDiff(a, b))


def tapeIsSame(a: float, b: float) -> bool:
    """
    True if given tape readings are approximately the same.

    Args:
      a: Distance in meters
      b: Distance in meters
    """
    if a < 2:
        if 0.8 < (a / b) < 1.2:
            return True
        return abs(a - b) < 0.05
    return 0.9 < (a / b) < 1.1


def shotIsSame(lhs: DataDict, rhs: DataDict) -> bool:
    """
    True if given shots are approximately the same.

    Args:
      lhs: "Left-hand-side" shot to compare (cannot be empty)
      rhs: "Right-hand-side" shot to compare (can be empty dict)
    """
    return bool(rhs and rhs[Column.TAPE]
                and tapeIsSame(lhs[Column.TAPE], rhs[Column.TAPE])
                and absAngleDiff(lhs[Column.CLINO], rhs[Column.CLINO]) < 2
                and absAngleDiff(lhs[Column.COMPASS], rhs[Column.COMPASS]) < 2)


class FindDuplicateFormatter(Formatter):
    def __init__(self, file=None):
        self._prev = {}
        self._count_consecutive = 0
        self._count_consecutive_splays = 0
        self._count_consecutive_legs = 0
        super().__init__(file)

    def skipAnonymousStations(self):
        return False

    def printDataLine(self, data):
        if shotIsSame(data, self._prev):
            self._count_consecutive += 1
            is_anon = (
                data[Column.TO] in ANONYMOUS_STATIONS,
                self._prev[Column.TO] in ANONYMOUS_STATIONS,
            )
            same_leg = (data[Column.FROM] == self._prev[Column.FROM]
                        and data[Column.TO] == self._prev[Column.TO])
            if all(is_anon):
                self._count_consecutive_splays += 1
                indicator = f"{self._count_consecutive_splays:2d}/"
            elif any(is_anon):
                self._count_consecutive_splays = 0
                indicator = "   "
            elif same_leg:
                self._count_consecutive_splays = 0
                indicator = " D "
            else:
                self._count_consecutive_legs += 1
                self._count_consecutive_splays = 0
                indicator = " L "
            if self._count_consecutive < 2:
                print("prev  :", self._prev, file=self.file)
            print(f" {indicator}{self._count_consecutive:2d}:", data, file=self.file)
        else:
            self._reset_consecutive()
        self._prev = data

    def printFooter(self):
        self._reset_consecutive()

    def _reset_consecutive(self):
        if self._count_consecutive_splays >= 2:
            print(f" check: leg? {self._prev[Column._FILE]}", file=self.file)
        if self._count_consecutive_legs >= 1:
            print(f" check: dup? {self._prev[Column._FILE]}", file=self.file)
        self._count_consecutive = 0
        self._count_consecutive_legs = 0
        self._count_consecutive_splays = 0


class VisualTopoFormatter(Formatter):
    declination = 0
    date = None, None, None
    param_line_pending = True
    colormask = 0b000

    def printHeader(self):
        print("""Version 5.11
Verification 1

Trou XXX,540900.0,5362363.0,UTM32
Club XXX
Toporobot 0
Couleur 0,0,0
""",
              end='',
              file=self.file)
        if self.entrances:
            entrance = self.normalizeName(self.entrances[0])
            print(f"Entree {entrance}", file=self.file)

    def getRgb(self) -> tuple:
        self.colormask += 1
        if self.colormask == 0b111:
            self.colormask = 1
        return tuple(((self.colormask >> s) & 1) * 0xFF for s in (0, 1, 2))

    def printPendingParamLine(self):
        if not self.param_line_pending:
            return

        self.param_line_pending = False

        def fmt(v, width):
            return '-' * width if v is None else v.zfill(width)

        year, month, day = self.date
        red, green, blue = self.getRgb()

        print(file=self.file)
        print(
            f"Param Deca Degd Clino Degd {self.declination:.4f} Dir,Dir,Dir Arr "
            f"{red},{green},{blue} {fmt(day, 2)}/{fmt(month, 2)}/{fmt(year, 4)} A",
            file=self.file)
        print(file=self.file)

    def printDate(self, data):
        self.param_line_pending = True
        self.date = data[Column.DATE]
        assert len(self.date) == 3

    def printDeclination(self, data):
        self.param_line_pending = True
        self.declination = data[Column.DECLINATION]

    def normalizeName(self, name: str, prefix: str = "_") -> str:
        if len(name) < VT_MAXLEN and not name.startswith(prefix):
            return name
        num = len(self._name_mapping)
        num = self._name_mapping.setdefault(name, num)
        unique = f"{prefix}{num}"
        n = VT_MAXLEN - len(unique) - 2
        if n > 0:
            return unique + "_" + name[-n:]
        return unique

    def printDataLine(self, data):
        station_from = self.normalizeName(data[Column.FROM])
        flags = data.get(Column.FLAGS, {})

        if data[Column.TO] in ANONYMOUS_STATIONS:
            station_to = "*"
            suffix = "E M *"
        else:
            station_to = self.normalizeName(data[Column.TO])
            if flags.get("surface") or flags.get("duplicate"):
                suffix = "E * N"
            else:
                suffix = "I * N"

        assert len(station_from) < VT_MAXLEN, station_from
        assert len(station_to) < VT_MAXLEN, station_to

        assert Column.CLINO in data, data

        self.printPendingParamLine()
        print(
            f"{station_from:11s}{station_to:22s}"
            f"{data[Column.TAPE]:8.2f}{data[Column.COMPASS]:8.2f}"
            f"{data[Column.CLINO]:8.2f}      *      *      *      * N {suffix}",
            file=self.file)


@dataclass
class LineInfo:
    path: Path
    lineno: int = 0

    def __str__(self) -> str:
        return self.path.as_posix() + f":{self.lineno}"


class VisualTopoXmlFormatter(VisualTopoFormatter):
    _param: Optional[etree.Element] = None

    def printHeader(self):
        self._root = etree.fromstring("""
<VisualTopo>
  <Version>5.23</Version>
  <Cavite>
    <Entrees/>
  </Cavite>
  <Mesures/>
  <Configuration/>
</VisualTopo>
""")
        self._mesures = self._root.find("Mesures")
        assert self._mesures is not None

    def printFooter(self):
        if self.entrances:
            node = self._root.find("Cavite/Entrees")
            assert node is not None
            etree.SubElement(
                node, "Entree", {
                    "X": "0.0",
                    "Y": "0.0",
                    "Z": "0.0",
                    "Station": self.entrances[0],
                    "Ref": "1",
                })

        etree.indent(self._root, space=" ")
        self.file.write('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n')
        self.file.write(etree.tostring(self._root, encoding="unicode"))

    def printPendingParamLine(self):
        if not self.param_line_pending:
            return

        self.param_line_pending = False

        year, month, day = self.date
        red, green, blue = self.getRgb()

        self._param = etree.SubElement(
            self._mesures, "Param", {
                "InstrDist": "Deca",
                "UnitDir": "Degd",
                "InstrPte": "Clino",
                "UnitPte": "Degd",
                "Declin": str(self.declination),
                "SensDir": "Dir",
                "SensPte": "Dir",
                "SensLar": "Dir",
                "DimPt": "Arr",
                "Coul": f"{red},{green},{blue}",
                "DeclinAuto": "M",  # A
            })

        if year is not None:
            self._param.set("Date", f"{day or '--'}/{month or '--'}/{year}")

    def insertOriginPoint(self, station: str):
        assert self._param is not None
        etree.SubElement(
            self._param, "Visee", {
                "Dep": station,
                "Arr": station,
                "Long": "0.0",
                "Az": "0.0",
                "Pte": "0.0",
            })

    def printDataLine(self, data):
        flags = data.get(Column.FLAGS, {})
        station_to = data[Column.TO]

        exclude = (
            station_to in ANONYMOUS_STATIONS  #
            or flags.get("surface")  #
            or flags.get("duplicate")  #
        )

        is_first = self._param is None

        self.printPendingParamLine()
        assert self._param is not None

        if is_first:
            # Without this, I get a "No origin point" error
            self.insertOriginPoint(data[Column.FROM])

        node = etree.SubElement(
            self._param, "Visee", {
                "Dep": data[Column.FROM],
                "Long": f"{data[Column.TAPE]}",
                "Az": f"{data[Column.COMPASS]}",
                "Pte": f"{data[Column.CLINO]}",
            })

        if station_to not in ANONYMOUS_STATIONS:
            node.set("Arr", station_to)

        if exclude:
            node.set("Exc", "E")


class SvxParser:
    """
    Parser for Survex data files (*.svx)
    """
    def __init__(self) -> None:
        self._path_stack: List[LineInfo] = []
        self._prefixes: List[str] = []
        self._alias_stack: List[Dict[str, str]] = [{}]
        self._units_stack = [DEFAULT_UNITS.copy()]
        self._calibrate_stack = [DEFAULT_CALIBRATE.copy()]
        self._case_stack = ["tolower"]
        self._data_style_stack: List[Tuple[Optional[DataStyle], List[Column]]] = [
            (DataStyle.NORMAL, DEFAULT_COLUMN_ORDER)
        ]
        self._data_order_pending: List[Column] = []
        self._data_table: List[DataDict] = []
        self._flags_stack: List[Dict[str, bool]] = [{}]
        self._entrance_set: set[str] = set()

    def iterdata(self):
        """
        Iterator over the data table.
        """
        return iter(self._data_table)

    @staticmethod
    def splitLine(line: str) -> list:
        """
        Split a line into tokens, discarding comments.
        """
        splitter = shlex.shlex(line)
        splitter.commenters = ';'
        splitter.quotes = '"'
        splitter.whitespace_split = True
        return list(splitter)

    def caseConvert(self, name: str) -> str:
        """
        Convert name according to the `*case` command.
        """
        if self._case_stack[-1] == 'tolower':
            return name.lower()
        if self._case_stack[-1] == 'toupper':
            return name.upper()
        return name

    def withPrefix(self, station: str) -> str:
        """
        Full name of the given station.
        """
        station = self._alias_stack[-1].get(station, station)
        station = self.caseConvert(station)

        if station in ANONYMOUS_STATIONS:
            return station

        return '.'.join(p for p in (self._prefixes + [station]) if p)

    @property
    def currentPath(self) -> Path:
        """
        Current .svx file path.
        """
        return self._path_stack[-1].path

    def formatLineInfo(self) -> str:
        """
        Current .svx file path and line number.
        """
        return str(self._path_stack[-1])

    def processLine(self, line: str):
        """
        Process one line of a .svx file.
        """
        tokens = list(self.splitLine(line))

        if not tokens or tokens == ['']:
            self._data_order_pending = []
            return

        if tokens[0].startswith('*'):
            self._data_order_pending = []
            self.processCommand(*tokens)
        else:
            self.processData(tokens)

    def parseFile(self, svxfile: "Path | str"):
        """
        Parse a .svx file
        """
        warn('parseFile', svxfile, level='Info')
        svxfile = Path(svxfile)

        lineinfo = LineInfo(svxfile)
        self._path_stack.append(lineinfo)

        with open(svxfile) as handle:
            for lineno, line in enumerate(handle, 1):
                lineinfo.lineno = lineno
                self.processLine(line.strip())

        self._path_stack.pop()

        return self

    def processCommand(self, command: str, *_args):
        """
        Process a command.
        """
        args = list(_args)

        assert command.startswith('*')
        command = args.pop(0) if command == '*' else command[1:]

        try:
            command = Command.from_string(command)
        except NoSuchEnumError as ex:
            warn(ex)
            return

        if command == Command.BEGIN:
            self.processBegin(*args)
        elif command == Command.END:
            self.processEnd(*args)
        elif command == Command.DATA:
            self.processDataHeader(*args)
        elif command == Command.INCLUDE:
            self.processInclude(*args)
        elif command == Command.EQUATE:
            self.processEquate(args)
        elif command == Command.CALIBRATE:
            self.processCalibrate(args)
        elif command == Command.UNITS:
            self.processUnits(args)
        elif command == Command.ALIAS:
            self.processAlias(args)
        elif command == Command.DATE:
            self.processDate(args)
        elif command == Command.DECLINATION:
            self.processDeclination(args)
        elif command == Command.ENTRANCE:
            self.processEntrance(args)
        elif command == Command.FLAGS:
            self.processFlags(args)
        elif command == Command.CASE:
            self.processCase(args)
        else:
            warn(command, 'not implemented', level='Info')

    def processBegin(self, prefix: str = ""):
        prefix = self.caseConvert(prefix)
        self._prefixes.append(prefix)
        self._data_style_stack.append(self._data_style_stack[-1])
        self._alias_stack.append(self._alias_stack[-1].copy())
        self._units_stack.append(self._units_stack[-1].copy())
        self._calibrate_stack.append(self._calibrate_stack[-1].copy())
        self._case_stack.append(self._case_stack[-1])
        self._flags_stack.append(self._flags_stack[-1].copy())

    def processEnd(self, prefix: str = ""):
        prefix_begin = self._prefixes.pop()
        if prefix and prefix.lower() != prefix_begin.lower():
            raise ValueError(
                f"'*end {prefix}' does not match '*begin {prefix_begin}'")
        self._data_style_stack.pop()
        self._alias_stack.pop()
        self._units_stack.pop()
        self._calibrate_stack.pop()
        self._case_stack.pop()
        self._flags_stack.pop()

    def processDataHeader(self, style: str, *ordering):
        """
        Process *DATA arguments. Unknown data styles are reported and skipped.
        """
        try:
            data_style = DataStyle.from_string(style)
        except NoSuchEnumError as ex:
            self._data_style_stack[-1] = (None, [])
            warn(ex)
            return

        if data_style == DataStyle.DEFAULT:
            self._data_style_stack[-1] = (DataStyle.NORMAL,
                                          DEFAULT_COLUMN_ORDER)
        else:
            self._data_style_stack[-1] = (data_style, [
                Column.from_string(col) for col in ordering
            ])

    def processData(self, tokens: List[str]):
        """
        Process data according the current data style.
        """
        data_style, data_order = self._data_style_stack[-1]

        if data_style not in (DataStyle.NORMAL, DataStyle.DIVING):
            return

        if self._data_order_pending:
            data_order = self._data_order_pending
            self._data_order_pending = []

        data: DataDict = dict(zip(data_order, tokens))

        pending = data_order[len(data):]
        if pending and pending[0] != Column.IGNOREALL:
            if pending[0] == Column.NEWLINE:
                self._data_order_pending = pending[1:]
            else:
                warn(f"Missing tokens for {pending}")

        for col in STATION_COLUMNS:
            if col in data:
                data[col] = self.withPrefix(data[col])

        for col in [Column.TAPE, Column.COMPASS, Column.CLINO, Column.DEPTH, Column.DEPTHCHANGE]:
            if col in data:
                try:
                    data[col] = NAMED_VALUES[data[col].upper()]
                    continue
                except KeyError:
                    pass

                reading = float(data[col])
                factor, unit = self._units_stack[-1][col]
                zero_error, zero_error_unit, scale = self._calibrate_stack[-1][
                    col]

                if zero_error_unit is None:
                    reading -= zero_error

                reading = convert_unit(reading, unit) * factor * scale

                if zero_error_unit is not None:
                    reading -= convert_unit(zero_error, zero_error_unit)

                data[col] = reading

        data[Column._FILE] = self.formatLineInfo()

        flags = self._flags_stack[-1]
        if flags:
            data[Column.FLAGS] = flags.copy()

        self._data_table.append(data)

    def processInclude(self, filename: str):
        filename = filename.replace("\\", "/")
        path = self._path_stack[-1].path.parent / filename
        path = findFile(path, case_sensitive=False)
        self.parseFile(path)

    def processEquate(self, tokens: List[str]):
        """
        Process *EQUATE arguments.
        """
        stations = [self.withPrefix(station) for station in tokens]
        for station in stations[1:]:
            self._data_table.append({
                Column.FROM: stations[0],
                Column.TO: station,
                Column.TAPE: 0,
                Column.COMPASS: 0,
                Column.CLINO: 0,
                Column._FILE: self.formatLineInfo(),
            })

    def processAlias(self, tokens: List[str]):
        """
        Process *ALIAS arguments.
        """
        assert tokens[0] == 'station'

        if len(tokens) > 2:
            self._alias_stack[-1][tokens[1]] = tokens[2]
        else:
            self._alias_stack[-1].pop(tokens[1], None)

    def processUnits(self, tokens: List[str]):
        """
        Process *UNITS arguments.
        """
        if len(tokens) == 1 and tokens[0].lower() == 'default':
            self._units_stack[-1] = DEFAULT_UNITS.copy()
            return

        tokens = list(tokens)
        unit = Units.from_string(tokens.pop())

        factor = float(tokens.pop()) if tokens[-1][-1].isdigit() else 1

        for quantitytok in tokens:
            quantity = Column.from_string(quantitytok)
            self._units_stack[-1][quantity] = (factor, unit)

    def processDate(self, tokens: List[str]):
        """
        Process *DATE arguments.
        """
        datefrom, _, dateto = tokens[0].partition("-")
        year, month, day = (datefrom.split(".") + [None, None])[:3]
        self._data_table.append({Column.DATE: (year, month, day)})

    def processDeclination(self, tokens: List[str]):
        """
        Process *DECLINATION arguments.
        """
        if tokens[0].lower() == "auto":
            warn('*declination auto not implemented', level='Info')
            return

        declination = float(tokens[0])
        assert tokens[1].lower().startswith("deg")
        self._data_table.append({Column.DECLINATION: declination})

    def processEntrance(self, tokens: List[str]):
        """
        Process *ENTRANCE arguments.
        """
        station = self.withPrefix(tokens[0])
        self._entrance_set.add(station)

    def processFlags(self, tokens: List[str]):
        """
        Process *FLAGS arguments.
        """
        value = True
        for tok in tokens:
            if tok == "not":
                value = False
            else:
                self._flags_stack[-1][tok] = value
                value = True

    def processCalibrate(self, tokens: List[str]):
        """
        Process *CALIBRATE arguments.
        """
        if len(tokens) == 1 and tokens[0].lower() == 'default':
            self._calibrate_stack[-1] = DEFAULT_CALIBRATE.copy()
            return

        quantities: List[Column] = []
        for quantitytok in tokens:
            try:
                quantities.append(Column.from_string(quantitytok))
            except NoSuchEnumError:
                break

        idx = len(quantities)

        zero_error = float(tokens[idx])
        idx += 1

        try:
            units = Units.from_string(tokens[len(quantities) + 1])
            idx += 1
        except (NoSuchEnumError, IndexError):
            units = None

        if idx < len(tokens):
            scale = float(tokens[idx])
            idx += 1
        else:
            scale = 1

        assert idx == len(tokens)

        for quantity in quantities:
            self._calibrate_stack[-1][quantity] = (zero_error, units, scale)

    def processCase(self, tokens: List[str]):
        """
        Process *CASE arguments.
        """
        value = tokens[0].lower()
        if value not in ["preserve", "toupper", "tolower"]:
            warn(f'Unknown *case {value}')
        self._case_stack[-1] = value

    def promoteDivingLegs(self):
        """
        Promote data readings between DIVING stations to NORMAL legs with FROM,
        TO and CLINO.
        """
        data_from: Optional[DataDict] = None
        data_leg: Optional[DataDict] = None

        for data in self._data_table:
            if Column.FROM in data:
                data_from = None
                data_leg = None
            elif Column.STATION in data:
                if data_from and data_leg:
                    data_leg[Column.FROM] = data_from[Column.STATION]
                    data_leg[Column.TO] = data[Column.STATION]
                    if Column.DEPTHCHANGE not in data_leg and Column.DEPTH in data:
                        assert Column.DEPTH in data_from
                        data_leg[Column.DEPTHCHANGE] = data[
                            Column.DEPTH] - data_from[Column.DEPTH]
                data_from = data
                data_leg = None
            elif Column.TAPE in data:
                assert Column.TO not in data
                assert data_leg is None
                data_leg = data

        for data in self._data_table:
            if Column.CLINO not in data and Column.DEPTHCHANGE in data:
                data[Column.CLINO] = math.degrees(
                    math.atan2(data[Column.DEPTHCHANGE],
                               data[Column.TAPE]))

    def dump(self, formatter: Formatter):
        """
        Dump the data with the given formatter.
        """
        formatter.setEntrances(self._entrance_set)
        formatter.printHeader()

        for data in self._data_table:
            if Column.DATE in data:
                formatter.printDate(data)

            if Column.DECLINATION in data:
                formatter.printDeclination(data)

            if Column.FROM not in data:
                continue

            if formatter.skipAnonymousStations() and any(
                    station in ANONYMOUS_STATIONS
                    for station in [data[Column.FROM], data[Column.TO]]):
                continue

            formatter.printDataLine(data)

        formatter.printFooter()


def main() -> None:
    parser = SvxParser()
    formatter: Formatter = SvxFormatter()

    for filename in sys.argv[1:]:
        if filename == '--help':
            print(f"usage: python {sys.argv[0]} [--spelix|--tro|--trox|--dups] <svxfiles...>")
            return

        if filename == '--spelix':
            formatter = SpelixCsvFormatter()
            continue

        if filename == '--tro':
            formatter = VisualTopoFormatter()
            continue

        if filename == '--trox':
            formatter = VisualTopoXmlFormatter()
            continue

        if filename == '--dups':
            formatter = FindDuplicateFormatter()
            continue

        parser.parseFile(findFile(Path(filename)))

    parser.promoteDivingLegs()
    parser.dump(formatter)


if __name__ == '__main__':
    main()
