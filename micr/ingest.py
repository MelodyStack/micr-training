"""Copy check photos into checks/ under clean, stable ids.

    python -m micr.ingest --from "D:\\photos\\checks"

Camera and messaging apps produce filenames like
``WhatsApp Image 2026-09-14 at 10.16.01 PM.jpeg``. The check id is the filename
stem, and labels.txt is whitespace-separated, so an id containing spaces
silently matches nothing -- every such check reports "NO LABEL" and contributes
no crops. This renames them to ``chk001``, ``chk002``, ... on the way in.

Originals are copied, never moved, unless --move is passed. The mapping back to
the original filename is recorded in ``checks/sources.csv`` so a bad crop can
always be traced to the photo it came from.

Re-running is safe: files already ingested are recognised by content hash and
skipped, so the same folder can be re-scanned after adding more photos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from .segment import CHECK_PATTERNS

SOURCES_NAME = "sources.csv"
SOURCE_FIELDS = ("check_id", "filename", "original_path", "sha256")
CHECK_SUFFIXES = tuple(pat.lstrip("*") for pat in CHECK_PATTERNS)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sources(checks_dir: Path) -> list[dict]:
    path = checks_dir / SOURCES_NAME
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_sources(checks_dir: Path, rows: list[dict]) -> None:
    path = checks_dir / SOURCES_NAME
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def next_index(existing: list[dict], prefix: str, checks_dir: Path) -> int:
    """First free index, counting both sources.csv and what is on disk.

    Photos can be in checks/ without a sources.csv row -- copied in by hand, or
    ingested before this tool existed. Numbering from sources.csv alone would
    reissue an id that is already taken and overwrite a real check.
    """
    candidates = [row["check_id"] for row in existing]
    candidates += [p.stem for p in checks_dir.iterdir() if p.is_file()]

    highest = 0
    for check_id in candidates:
        if check_id.startswith(prefix):
            suffix = check_id[len(prefix):]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return highest + 1


def find_images(source: Path, recursive: bool) -> list[Path]:
    globber = source.rglob if recursive else source.glob
    found = {p.resolve() for pat in CHECK_PATTERNS for p in globber(pat)}
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy check photos into checks/ under clean ids."
    )
    parser.add_argument("--from", dest="source", required=True, help="folder of photos")
    parser.add_argument("--checks-dir", default="checks")
    parser.add_argument("--prefix", default="chk")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--recursive", action="store_true", help="also scan subfolders")
    parser.add_argument("--move", action="store_true", help="move instead of copying")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"not a folder: {source}")

    checks_dir = Path(args.checks_dir)
    checks_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(source, args.recursive)
    if not images:
        raise SystemExit(
            f"no images in {source}"
            + ("" if args.recursive else " (try --recursive)")
        )

    rows = read_sources(checks_dir)
    known = {row["sha256"]: row["check_id"] for row in rows}
    index = next_index(rows, args.prefix, checks_dir)

    added: list[tuple[str, Path]] = []
    duplicates: list[tuple[Path, str]] = []

    for image in images:
        digest = file_hash(image)
        if digest in known:
            duplicates.append((image, known[digest]))
            continue

        check_id = f"{args.prefix}{index:0{args.digits}d}"
        filename = f"{check_id}{image.suffix.lower()}"
        destination = checks_dir / filename

        # Belt and braces: never write over an existing photo.
        while any(destination.with_suffix(s).exists() for s in CHECK_SUFFIXES):
            index += 1
            check_id = f"{args.prefix}{index:0{args.digits}d}"
            filename = f"{check_id}{image.suffix.lower()}"
            destination = checks_dir / filename

        if not args.dry_run:
            if args.move:
                shutil.move(str(image), destination)
            else:
                shutil.copy2(image, destination)
            rows.append(
                {
                    "check_id": check_id,
                    "filename": filename,
                    "original_path": str(image),
                    "sha256": digest,
                }
            )
        known[digest] = check_id
        added.append((check_id, image))
        index += 1

    for check_id, original in added:
        print(f"  {check_id}  <- {original.name}")
    if duplicates:
        print(f"\n{len(duplicates)} already ingested, skipped:")
        for original, check_id in duplicates[:5]:
            print(f"  {original.name} == {check_id}")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")

    if args.dry_run:
        print(f"\ndry run: {len(added)} would be added")
        return

    if added:
        write_sources(checks_dir, rows)
        labels_path = checks_dir / "labels.txt"
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n# --- {len(added)} check(s) added by micr.ingest ---\n"
                "# Remove the '#' and type the MICR line for each.\n"
            )
            for check_id, original in added:
                handle.write(f"# {check_id}  \t\t\t(from {original.name})\n")

    print(f"\n{len(added)} added, {len(duplicates)} skipped -> {checks_dir}")
    print(f"mapping back to originals: {checks_dir / SOURCES_NAME}")
    if added:
        print(
            f"\nNext: open {checks_dir / 'labels.txt'} and fill in the MICR line for\n"
            f"each of the {len(added)} new entries, then run:\n\n"
            "  python -m micr.build"
        )


if __name__ == "__main__":
    main()
