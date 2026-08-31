"""In-place notebook execution helper (the nbconvert CLI is broken on this machine).

Executes every cell of the given notebook via nbclient.NotebookClient after
installing the Windows selector event-loop policy (required for kernel
subprocesses here), saves the notebook in place WITH outputs, and exits
nonzero if any code cell raised.

Usage:
  python src/run_notebook.py "src/8. SCE Enrichment.ipynb"
  python src/run_notebook.py "src/7. Customer Enrichment.ipynb" --timeout 900
"""

import argparse
import asyncio
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Execute a repo notebook in place via nbclient."
    )
    parser.add_argument("notebook", type=Path, help="path to the .ipynb to execute")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-cell timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--kernel", default="python3", help="kernel name (default: python3)"
    )
    args = parser.parse_args(argv)

    nb_path = args.notebook.resolve()
    if not nb_path.exists():
        print("notebook not found: %s" % nb_path, file=sys.stderr)
        return 2

    # Must precede nbclient: kernel subprocesses need the selector loop policy.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError  # not re-exported at the root (nbclient 0.11)

    notebook = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        # Kernel cwd = the notebook's directory (src/), so ../data paths resolve.
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    raised = False
    try:
        client.execute()
    except CellExecutionError as exc:
        raised = True
        print(str(exc), file=sys.stderr)
    finally:
        nbformat.write(notebook, nb_path)  # saved in place WITH outputs

    error_cells = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(o.get("output_type") == "error" for o in cell.get("outputs", []))
    ]
    if error_cells:
        print("cells raised: %s" % error_cells, file=sys.stderr)
        return 1
    if raised:
        return 1
    n_code = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    print("executed ok: %s (%d code cells)" % (nb_path, n_code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
