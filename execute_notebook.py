"""Execute with a temporary kernel pointing at the current Python environment."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main() -> None:
    root = Path(__file__).resolve().parent
    path = root / "fastapi-llm-agent-case-study.ipynb"
    notebook = nbformat.read(path, as_version=4)
    with TemporaryDirectory() as tmp:
        kernel = Path(tmp) / "case-study"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Case study", "language": "python",
            "env": {"IPYTHONDIR": str(Path(tmp) / "ipython"),
                    "MPLCONFIGDIR": str(Path(tmp) / "matplotlib")},
        }), encoding="utf-8")
        manager = KernelManager(kernel_name="case-study",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
        try:
            NotebookClient(notebook, km=manager, timeout=120,
                           resources={"metadata": {"path": str(root)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells successfully.")


if __name__ == "__main__":
    main()
