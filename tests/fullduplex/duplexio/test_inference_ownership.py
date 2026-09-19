"""Serving must not import the application that uses it for training rollouts."""

import ast
from pathlib import Path


def test_inference_has_no_training_application_dependency():
    root = Path(__file__).resolve().parents[3] / "vllm_omni"
    dependencies = []
    for directory in (root / "experimental/fullduplex/duplexio", root / "model_executor/models/duplexio"):
        for path in directory.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if module.split(".")[0] == "duplexio":
                        dependencies.append(f"{path.relative_to(root)}:{node.lineno}: {module}")
    assert not dependencies, "Inference imports training application code:\n" + "\n".join(dependencies)
