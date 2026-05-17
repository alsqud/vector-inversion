from pathlib import Path


def load_lines(path):
    """
    Load non-empty lines from a text file.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return lines


def split_term_definition(line):
    """
    Split one line into term and definition.

    Expected format:
        term: definition

    Only the first colon is used as the separator.
    """
    line = str(line).strip()

    if ":" in line:
        term, definition = line.split(":", 1)
        return term.strip(), definition.strip()

    return line[:30].strip(), line.strip()


def load_term_definition_file(path):
    """
    Load a term-definition file and return structured fields.
    """
    lines = load_lines(path)

    terms = []
    definitions = []
    term_defs = []

    for line in lines:
        term, definition = split_term_definition(line)

        terms.append(term)
        definitions.append(definition)
        term_defs.append(f"{term}: {definition}")

    return {
        "lines": lines,
        "terms": terms,
        "definitions": definitions,
        "term_defs": term_defs,
    }


def load_dataset(data_dir):
    """
    Load train, validation, and test files from a dataset directory.
    """
    data_dir = Path(data_dir)

    train = load_term_definition_file(data_dir / "train.txt")
    val = load_term_definition_file(data_dir / "val.txt")
    test = load_term_definition_file(data_dir / "test.txt")

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def check_dataset(data_dir):
    """
    Check whether required dataset files exist and return basic statistics.
    """
    data_dir = Path(data_dir)

    required_files = ["train.txt", "val.txt", "test.txt"]

    for filename in required_files:
        file_path = data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")

    dataset = load_dataset(data_dir)

    return {
        "train_size": len(dataset["train"]["lines"]),
        "val_size": len(dataset["val"]["lines"]),
        "test_size": len(dataset["test"]["lines"]),
        "test_sample": dataset["test"]["lines"][:3],
    }