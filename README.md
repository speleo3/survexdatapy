[![CI](https://github.com/speleo3/survexdatapy/actions/workflows/ci.yml/badge.svg)](https://github.com/speleo3/survexdatapy/actions/workflows/ci.yml)

# survexdata

Python module for reading [Survex](https://survex.com/)
[data files](https://survex.com/docs/manual/datafile.htm).

## Usage

### Reading files

Read one or more files by calling `parseFile` for each file:

```python
parser = SvxParser()
parser.parseFile("example.svx")
```

### Writing files


There are several `Formatter` classes for writing the previously parsed data
to a new file, e.g. to a CSV file for import into Spelix:

```python
with open("spelix.csv", "w") as handle:
    parser.dump(SpelixCsvFormatter(handle))
```

### Accessing data

After reading a file, the data table can be iterated over with `iterdata`:

```python
print("Total length:", sum(leg[Column.TAPE] for leg in parser.iterdata()))
```

### CLI usage

Command line usage to convert to a supported format (to STDOUT):

```bash
python -m survexdata example.svx --csv
```

## License

BSD-2-Clause (c) Thomas Holder
