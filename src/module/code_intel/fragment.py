from __future__ import annotations

from src.module.code_intel.models import (
    CodeEdge,
    CodeFile,
    CodeFragment,
    CodeFragmentEdge,
    CodeFragmentEdgeKind,
    CodeFragmentNode,
    CodeFragmentNodeKind,
    CodeFragmentSource,
    RepositoryRecord,
    Symbol,
)


def repository_node_id(repo_id: str) -> str:
    return f"code_intel:repository:{repo_id}"


def file_node_id(repo_id: str, path: str) -> str:
    return f"code_intel:file:{repo_id}:{path}"


def symbol_node_id(symbol_id: str) -> str:
    return f"code_intel:symbol:{symbol_id}"


def build_code_fragment(
    *,
    repository: RepositoryRecord,
    files: list[CodeFile],
    symbols: list[Symbol],
    edges: list[CodeEdge],
) -> CodeFragment:
    repo_node_id = repository_node_id(repository.repo_id)
    sorted_files = sorted(files, key=lambda item: item.path)
    sorted_symbols = sorted(
        symbols,
        key=lambda item: (item.file_path, item.start_line, item.qualified_name),
    )
    sorted_edges = sorted(
        edges,
        key=lambda item: (item.file_path, item.line or 0, item.kind.value, item.target_name),
    )

    nodes = [
        CodeFragmentNode(
            node_id=repo_node_id,
            kind=CodeFragmentNodeKind.REPOSITORY,
            label=repository.label,
            properties={
                "repo_id": repository.repo_id,
                "root_path": repository.root_path,
                "git_head": repository.git_head,
            },
        )
    ]
    nodes.extend(_file_node(file) for file in sorted_files)
    nodes.extend(_symbol_node(symbol) for symbol in sorted_symbols)

    fragment_edges = [
        CodeFragmentEdge(
            edge_id=f"code_intel:contains:{repository.repo_id}:{file.path}",
            kind=CodeFragmentEdgeKind.CONTAINS,
            source_node_id=repo_node_id,
            target_node_id=file_node_id(file.repo_id, file.path),
            reason="repository contains indexed file",
        )
        for file in sorted_files
    ]
    fragment_edges.extend(
        CodeFragmentEdge(
            edge_id=f"code_intel:defines:{symbol.symbol_id}",
            kind=CodeFragmentEdgeKind.DEFINES,
            source_node_id=file_node_id(symbol.repo_id, symbol.file_path),
            target_node_id=symbol_node_id(symbol.symbol_id),
            reason="file defines symbol",
            properties={
                "file_path": symbol.file_path,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
            },
        )
        for symbol in sorted_symbols
    )

    target_by_name = _unique_symbol_targets(sorted_symbols)
    fragment_edges.extend(_code_edge(edge, target_by_name) for edge in sorted_edges)

    return CodeFragment(
        fragment_id=f"code_intel:fragment:{repository.repo_id}:{repository.indexed_at.isoformat()}",
        repo_id=repository.repo_id,
        source=CodeFragmentSource(
            repo_id=repository.repo_id,
            root_path=repository.root_path,
            git_head=repository.git_head,
            indexed_at=repository.indexed_at,
        ),
        nodes=nodes,
        edges=fragment_edges,
        summary={
            "file_count": len(files),
            "symbol_count": len(symbols),
            "edge_count": len(edges),
            "node_count": len(nodes),
            "fragment_edge_count": len(fragment_edges),
        },
    )


def _file_node(file: CodeFile) -> CodeFragmentNode:
    return CodeFragmentNode(
        node_id=file_node_id(file.repo_id, file.path),
        kind=CodeFragmentNodeKind.FILE,
        label=file.path,
        source_path=file.path,
        properties={
            "file_id": file.file_id,
            "language": file.language,
            "size_bytes": file.size_bytes,
            "sha256": file.sha256,
        },
    )


def _symbol_node(symbol: Symbol) -> CodeFragmentNode:
    return CodeFragmentNode(
        node_id=symbol_node_id(symbol.symbol_id),
        kind=CodeFragmentNodeKind.SYMBOL,
        label=symbol.qualified_name,
        source_path=symbol.file_path,
        properties={
            "symbol_id": symbol.symbol_id,
            "kind": symbol.kind.value,
            "name": symbol.name,
            "qualified_name": symbol.qualified_name,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "parent_symbol_id": symbol.parent_symbol_id,
            "is_async": symbol.is_async,
            "decorators": list(symbol.decorators),
        },
    )


def _code_edge(
    edge: CodeEdge,
    target_by_name: dict[str, str],
) -> CodeFragmentEdge:
    kind = (
        CodeFragmentEdgeKind.IMPORTS
        if edge.kind.value == CodeFragmentEdgeKind.IMPORTS.value
        else CodeFragmentEdgeKind.CALLS
    )
    return CodeFragmentEdge(
        edge_id=f"code_intel:edge:{edge.edge_id}",
        kind=kind,
        source_node_id=(
            symbol_node_id(edge.source_symbol_id)
            if edge.source_symbol_id
            else file_node_id(edge.repo_id, edge.file_path)
        ),
        target_node_id=target_by_name.get(edge.target_name),
        target_name=edge.target_name,
        reason="indexed code edge",
        properties={
            "file_path": edge.file_path,
            "line": edge.line,
        },
    )


def _unique_symbol_targets(symbols: list[Symbol]) -> dict[str, str]:
    buckets: dict[str, list[str]] = {}
    for symbol in symbols:
        node_id = symbol_node_id(symbol.symbol_id)
        buckets.setdefault(symbol.qualified_name, []).append(node_id)
        buckets.setdefault(symbol.name, []).append(node_id)
    return {
        name: node_ids[0]
        for name, node_ids in buckets.items()
        if len(node_ids) == 1
    }
