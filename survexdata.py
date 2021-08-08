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
from enum import Enum, auto


class UpperCaseStringEnum(Enum):
    """
    Base class for enums with upper case keys and (key == value).
    """
    def _generate_next_value_(name, start, count, last_values):
        assert name == name.upper()
        return name

    @classmethod
    def from_string(cls, value):
        value = value.upper()
        return cls(cls.aliases().get(value, value))

    @classmethod
    def aliases(cls) -> dict:
        return {}


class Command(UpperCaseStringEnum):
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
    ALIAS = auto()
    REQUIRE = auto()


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
    CLINO = auto()
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

    @classmethod
    def aliases(cls):
        return {
            "LENGTH": "TAPE",
            "BEARING": "COMPASS",
            "GRADIENT": "CLINO",
            "COUNT": "COUNTER",
        }


class Units(UpperCaseStringEnum):
    YARDS = auto()
    FEET = auto()
    METRES = auto()
    DEGREES = auto()
    GRADS = auto()
    MINUTES = auto()
    PERCENT = auto()
    QUADS = auto()

    @classmethod
    def aliases(cls):
        return {
            "METRIC": "METRES",
            "METERS": "METRES",
            "DEGS": "DEGREES",
            "PERCENTAGE": "PERCENT",
            "QUADRANTS": "QUADS",
        }


DEFAULT_COLUMN_ORDER = [
    Column.FROM,
    Column.TO,
    Column.TAPE,
    Column.COMPASS,
    Column.CLINO,
]

DEFAULT_CALIBRATE = {
    Column.TAPE: (0, Units.METRES, 1),
    Column.COMPASS: (0, Units.DEGREES, 1),
    Column.CLINO: (0, Units.DEGREES, 1),
    Column.DECLINATION: (0, Units.DEGREES, 1),
}

DEFAULT_UNITS = {
    Column.TAPE: (1, Units.METRES),
    Column.COMPASS: (1, Units.DEGREES),
    Column.CLINO: (1, Units.DEGREES),
    Column.DECLINATION: (1, Units.DEGREES),
}

UNIT_CONVERT = {
    Units.METRES: 1,
    Units.FEET: 0.3048,
    Units.DEGREES: 1,
    Units.GRADS: 0.9,
}

NAMED_VALUES = {
    'DOWN': -90,
    'UP': 90,
    '-': 0,
    'LEVEL': 0,
}

ANONYMOUS_STATIONS = {'.', '..', '...'}


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


class Formatter:
    """
    Base class for dumping Survex data.
    """
    def __init__(self, file=None):
        self.file = sys.stdout if file is None else file
        self._survey_mapping = {}
        self._name_mapping = {}

    def normalizeName(self, name: str) -> str:
        return str(
            self._name_mapping.setdefault(name.lower(),
                                          len(self._name_mapping)))

    def normalizeStationName(self, name: str) -> str:
        survey, dot, station = name.rpartition('.')
        if not survey:
            return name

        survey = self._survey_mapping.setdefault(survey,
                                                 len(self._survey_mapping))

        return f"{survey}{dot}{station}"

    def skipAnonymousStations(self) -> bool:
        return False

    def printHeader(self):
        pass

    def printFooter(self):
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
        print('"Von";"Bis";"Länge";"Neigung";"Richtung";"L";"R";"O";"U";"RV"',
              file=self.file)

    def printDataLine(self, data):
        print(
            ';'.join(f'"{data[col].lower()}"'
                     for col in [Column.FROM, Column.TO]) + ';' +
            ';'.join(f'"{data[col]}"'
                     for col in [Column.TAPE, Column.CLINO, Column.COMPASS]) +
            ';"0"' * 5,
            file=self.file)


class VisualTopoFormatter(Formatter):
    def printHeader(self):
        entrance = '0'

        print(f"""Version 5.11
Verification 1

Trou XXX,540900.0,5362363.0,UTM32
Club XXX
Entree {entrance}
Toporobot 0
Couleur 0,0,0

Param Deca Degd Clino Degd 1.2048 Dir,Dir,Dir Arr 0,0,255 01/01/2021 A

{entrance:11s}{entrance:22s}    0.00    0.00    0.00   0.00   0.00   0.00   0.00 N I * N
""",
              end='',
              file=self.file)

    def printDataLine(self, data):
        station_from = self.normalizeName(data[Column.FROM])

        if data[Column.TO] in ANONYMOUS_STATIONS:
            station_to = "*"
            suffix = "E M *"
        else:
            station_to = self.normalizeName(data[Column.TO])
            suffix = "I * N"

        assert len(station_from) < 12
        assert len(station_to) < 12

        print(
            f"{station_from:11s}{station_to:22s}"
            f"{data[Column.TAPE]:8.2f}{data[Column.COMPASS]:8.2f}"
            f"{data[Column.CLINO]:8.2f}      *      *      *      * N {suffix}",
            file=self.file)


class SvxParser:
    """
    Parser for Survex data files (*.svx)
    """
    def __init__(self):
        self._path_stack = []
        self._prefixes = []
        self._alias_stack = [{}]
        self._units_stack = [DEFAULT_UNITS.copy()]
        self._calibrate_stack = [DEFAULT_CALIBRATE.copy()]
        self._data_style_stack = [(DataStyle.NORMAL, DEFAULT_COLUMN_ORDER)]
        self._data_table = []

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

    def withPrefix(self, station: str) -> str:
        """
        Full name of the given station.
        """
        station = self._alias_stack[-1].get(station, station)

        if station in ANONYMOUS_STATIONS:
            return station

        return '.'.join(p for p in (self._prefixes + [station]) if p)

    @property
    def currentPath(self) -> Path:
        """
        Current .svx file path.
        """
        return self._path_stack[-1]

    def processLine(self, line: str):
        """
        Process one line of a .svx file.
        """
        tokens = list(self.splitLine(line))

        if not tokens or tokens == ['']:
            return

        if tokens[0].startswith('*'):
            self.processCommand(*tokens)
        else:
            self.processData(tokens)

    def parseFile(self, svxfile: Path):
        """
        Parse a .svx file, given the context of previously parsed files.
        """
        warn('parseFile', svxfile, level='Info')
        svxfile = Path(svxfile)

        if svxfile.suffix != '.svx':
            svxfile = svxfile.with_suffix(svxfile.suffix + '.svx')

        if not svxfile.exists():
            pattern = re.sub(
                r'[a-zA-Z]',
                lambda c: f'[{c.group(0).lower()}{c.group(0).upper()}]',
                str(svxfile))
            svxfile = next(Path().glob(pattern))

        self._path_stack.append(svxfile)

        with open(svxfile) as handle:
            for line in handle:
                self.processLine(line.strip())

        self._path_stack.pop()

        return self

    def processCommand(self, command: str, *args):
        """
        Process a command.
        """
        args = list(args)

        assert command.startswith('*')
        command = args.pop(0) if command == '*' else command[1:]

        try:
            command = Command.from_string(command)
        except ValueError as ex:
            warn(ex)
            return

        if command == Command.BEGIN:
            assert len(args) <= 1
            self._prefixes.append(args[0] if args else None)
            self._data_style_stack.append(self._data_style_stack[-1])
            self._alias_stack.append(self._alias_stack[-1].copy())
            self._units_stack.append(self._units_stack[-1].copy())
            self._calibrate_stack.append(self._calibrate_stack[-1].copy())
        elif command == Command.END:
            self._prefixes.pop()
            self._data_style_stack.pop()
            self._alias_stack.pop()
            self._units_stack.pop()
            self._calibrate_stack.pop()
        elif command == Command.DATA:
            self.processDataHeader(*args)
        elif command == Command.INCLUDE:
            assert len(args) == 1
            self.parseFile(self.currentPath.parent / args[0])
        elif command == Command.EQUATE:
            self.processEquate(args)
        elif command == Command.CALIBRATE:
            self.processCalibrate(args)
        elif command == Command.UNITS:
            self.processUnits(args)
        elif command == Command.ALIAS:
            self.processAlias(args)

    def processDataHeader(self, style: str, *ordering):
        """
        Process *DATA arguments. Unknown data styles are reported and skipped.
        """
        try:
            data_style = DataStyle.from_string(style)
        except ValueError as ex:
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

    def processData(self, tokens: list):
        """
        Process data according the current data style.
        """
        data_style, data_order = self._data_style_stack[-1]

        if data_style != DataStyle.NORMAL:
            return

        data = dict(zip(data_order, tokens))

        for col in [Column.FROM, Column.TO, Column.STATION]:
            if col in data:
                data[col] = self.withPrefix(data[col])

        for col in [Column.TAPE, Column.COMPASS, Column.CLINO]:
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

        self._data_table.append(data)

    def processEquate(self, tokens: list):
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
            })

    def processAlias(self, tokens: list):
        """
        Process *ALIAS arguments.
        """
        assert tokens[0] == 'station'

        if len(tokens) > 2:
            self._alias_stack[-1][tokens[1]] = tokens[2]
        else:
            self._alias_stack[-1].pop(tokens[1], None)

    def processUnits(self, tokens: list):
        """
        Process *UNITS arguments.
        """
        if len(tokens) == 1 and tokens[0].lower() == 'default':
            self._units_stack[-1] = DEFAULT_UNITS.copy()
            return

        tokens = list(tokens)
        unit = tokens.pop()
        unit = Units.from_string(unit)

        factor = float(tokens.pop()) if tokens[-1][-1].isdigit() else 1

        for quantity in tokens:
            quantity = Column.from_string(quantity)
            self._units_stack[-1][quantity] = (factor, unit)

    def processCalibrate(self, tokens: list):
        """
        Process *CALIBRATE arguments.
        """
        if len(tokens) == 1 and tokens[0].lower() == 'default':
            self._calibrate_stack[-1] = DEFAULT_CALIBRATE.copy()
            return

        quantities = []
        for quantity in tokens:
            try:
                quantities.append(Column.from_string(quantity))
            except ValueError:
                break

        idx = len(quantities)

        zero_error = float(tokens[idx])
        idx += 1

        try:
            units = Units.from_string(tokens[len(quantities) + 1])
            idx += 1
        except (ValueError, IndexError):
            units = None

        if idx < len(tokens):
            scale = float(tokens[idx])
            idx += 1
        else:
            scale = 1

        assert idx == len(tokens)

        for quantity in quantities:
            self._calibrate_stack[-1][quantity] = (zero_error, units, scale)

    def dump(self, formatter: Formatter = SvxFormatter()):
        """
        Dump the data with the given formatter.
        """
        formatter.printHeader()

        for data in self.iterdata():
            if formatter.skipAnonymousStations() and any(
                    station in ANONYMOUS_STATIONS
                    for station in [data[Column.FROM], data[Column.TO]]):
                continue

            formatter.printDataLine(data)

        formatter.printFooter()


def main():
    parser = SvxParser()
    formatter = SvxFormatter()

    for filename in sys.argv[1:]:
        if filename == '--spelix':
            formatter = SpelixCsvFormatter()
            continue

        if filename == '--tro':
            formatter = VisualTopoFormatter()
            continue

        parser.parseFile(filename)

    parser.dump(formatter)


if __name__ == '__main__':
    main()
