"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype
from collections import defaultdict
from PIL import Image

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files
# these files are treated as image
image_files = {".png", ".jpeg", ".jpg", ".tiff"}

def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if f.suffix in plaintext_files:
        num_lines = sum(1 for _ in open(f))
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
        img_size_info = ""
        if f.suffix in image_files:
            try:
                with Image.open(f) as img:
                    width, height = img.size
                    img_size_info = f"{width}x{height} pixels"
            except:
                img_size_info = ""
        return s, str(humanize.naturalsize(s)) + f", {img_size_info}" 


def file_tree(path: Path, depth=0) -> str:
    """Generate a tree structure of files in a directory"""
    result = []
    files = [p for p in Path(path).iterdir() if not p.is_dir()]
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs):
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)


def _walk(path: Path):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    df = pd.read_csv(p)

    out = []

    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    """Generate a textual preview of a json file using a generated json schema"""
    builder = SchemaBuilder()
    with open(p) as f:
        builder.add_object(json.load(f))
    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
        indent=2
    )


def preview_xlsx(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of an Excel file

    Args:
        p (Path): the path to the Excel file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    # Read the Excel file
    df = pd.read_excel(p)

    out = []

    # Basic information about the file
    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        # Display a subset of columns
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        # Detailed information about each column
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_image(image_path, file_name):
    """
    Generate a preview for image files, including their size information.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return f"-> {file_name} is an image with size: {width}x{height} pixels"
    except Exception as e:
        return f"-> {file_name} could not be processed as an image: {str(e)}"
    

def generate(base_path, include_file_details=True, simple=False):
    """
    Generate a directory preview with aggregated image size information.
    Summarizes images by type, count, and size ranges instead of individual entries.
    """
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]

    if include_file_details:
        image_data = defaultdict(list)  # {suffix: [(width, height)]}
        image_errors = 0

        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix == ".xlsx":
                out.append(preview_xlsx(fn, file_name, simple=simple))
            elif fn.suffix in image_files:
                try:
                    with Image.open(fn) as img:
                        image_data[fn.suffix].append(img.size)
                except Exception:
                    image_errors += 1
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

        # Build image summary
        if image_data or image_errors:
            summary = ["Image Summary:"]
            
            for ext, sizes in image_data.items():
                count = len(sizes)
                ext_name = ext.lstrip('.').upper()
                areas = [w*h for w, h in sizes]
                min_size = sizes[areas.index(min(areas))]
                max_size = sizes[areas.index(max(areas))]
                
                size_range = (f"{min_size[0]}x{min_size[1]} - "
                            f"{max_size[0]}x{max_size[1]}")
                summary.append(f"  • {count} {ext_name}: {size_range}")
            
            if image_errors:
                summary.append(f"  • {image_errors} files could not be read")
            
            out.append("\n".join(summary))

    result = "\n\n".join(out)

    if len(result) > 6000 and not simple:
        return generate(base_path, include_file_details=include_file_details, simple=True)
    
    return result


if __name__ == "__main__":
    prev = generate(Path("/root/Med-DL-Agent/meddle/med_dl_tasks/odir5k_2d_mlc"))
    print(prev)