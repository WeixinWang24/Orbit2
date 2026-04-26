from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from src.module.code_intel.models import (
    CodeEdge,
    CodeFile,
    Diagnostic,
    EdgeKind,
    FileAnalysis,
    Symbol,
    SymbolKind,
)


def _stable_id(*parts: object) -> str:
    raw = "\0".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class PythonAstAnalyzer:
    language = "python"

    def analyze_file(self, *, repo_id: str, root: Path, path: Path) -> FileAnalysis:
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        code_file = CodeFile(
            file_id=_stable_id(repo_id, relative),
            repo_id=repo_id,
            path=relative,
            language=self.language,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        try:
            text = raw.decode("utf-8")
            tree = ast.parse(text, filename=relative)
        except SyntaxError as exc:
            return FileAnalysis(
                file=code_file,
                diagnostics=[
                    Diagnostic(
                        repo_id=repo_id,
                        file_path=relative,
                        severity="error",
                        message=exc.msg,
                        line=exc.lineno,
                    )
                ],
            )
        except UnicodeDecodeError as exc:
            return FileAnalysis(
                file=code_file,
                diagnostics=[
                    Diagnostic(
                        repo_id=repo_id,
                        file_path=relative,
                        severity="error",
                        message=str(exc),
                    )
                ],
            )

        visitor = _PythonAstVisitor(repo_id=repo_id, file_path=relative)
        visitor.visit(tree)
        return FileAnalysis(file=code_file, symbols=visitor.symbols, edges=visitor.edges)


class _PythonAstVisitor(ast.NodeVisitor):
    def __init__(self, *, repo_id: str, file_path: str) -> None:
        self.repo_id = repo_id
        self.file_path = file_path
        module_name = (
            file_path[:-3].replace("/", ".")
            if file_path.endswith(".py")
            else file_path
        )
        self.module_symbol = Symbol(
            symbol_id=_stable_id(repo_id, file_path, "module", module_name, 1),
            repo_id=repo_id,
            file_path=file_path,
            kind=SymbolKind.MODULE,
            name=module_name.rsplit(".", 1)[-1],
            qualified_name=module_name,
            start_line=1,
            end_line=1,
        )
        self.symbols: list[Symbol] = [self.module_symbol]
        self.edges: list[CodeEdge] = []
        self._stack: list[Symbol] = [self.module_symbol]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_edge(EdgeKind.IMPORTS, alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            target = f"{module}.{alias.name}" if module else alias.name
            self._add_edge(EdgeKind.IMPORTS, target, node.lineno)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol = self._symbol_for_node(node, SymbolKind.CLASS)
        self.symbols.append(symbol)
        self._stack.append(symbol)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def visit_Call(self, node: ast.Call) -> None:
        target = self._call_target(node.func)
        if target is not None:
            self._add_edge(EdgeKind.CALLS, target, node.lineno)
        self.generic_visit(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool,
    ) -> None:
        kind = (
            SymbolKind.METHOD
            if self._stack[-1].kind == SymbolKind.CLASS
            else SymbolKind.FUNCTION
        )
        symbol = self._symbol_for_node(node, kind, is_async=is_async)
        self.symbols.append(symbol)
        self._stack.append(symbol)
        self.generic_visit(node)
        self._stack.pop()

    def _symbol_for_node(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: SymbolKind,
        *,
        is_async: bool = False,
    ) -> Symbol:
        parent = self._stack[-1]
        qualified_name = f"{parent.qualified_name}.{node.name}"
        decorators = [self._expr_name(d) for d in node.decorator_list]
        decorators = [d for d in decorators if d is not None]
        end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
        symbol_id = _stable_id(
            self.repo_id,
            self.file_path,
            kind.value,
            qualified_name,
            node.lineno,
        )
        return Symbol(
            symbol_id=symbol_id,
            repo_id=self.repo_id,
            file_path=self.file_path,
            kind=kind,
            name=node.name,
            qualified_name=qualified_name,
            start_line=node.lineno,
            end_line=end_line,
            parent_symbol_id=parent.symbol_id,
            is_async=is_async,
            decorators=decorators,
        )

    def _add_edge(self, kind: EdgeKind, target_name: str, line: int | None) -> None:
        source = self._stack[-1]
        self.edges.append(
            CodeEdge(
                edge_id=_stable_id(
                    self.repo_id,
                    self.file_path,
                    kind.value,
                    source.symbol_id,
                    target_name,
                    line,
                    len(self.edges),
                ),
                repo_id=self.repo_id,
                file_path=self.file_path,
                kind=kind,
                source_symbol_id=source.symbol_id,
                target_name=target_name,
                line=line,
            )
        )

    def _call_target(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _expr_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            return self._call_target(node)
        return None
