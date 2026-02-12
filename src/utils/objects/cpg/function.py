from .ast import AST


class Function:
    def __init__(self, function):
        self.name = function.get("function", "")
        raw_id = function.get("id", "")
        self.id = str(raw_id).split(".")[-1] if raw_id else ""
        self.indentation = 1
        ast_nodes = function.get("AST")
        if ast_nodes is None:
            ast_nodes = self._normalize_ast(function)
        self.ast = AST(ast_nodes, self.indentation)

    def __str__(self):
        indentation = self.indentation * "\t"
        return f"{indentation}Function Name: {self.name}\n{indentation}Id: {self.id}\n{indentation}AST:{self.ast}"

    def get_nodes(self):
        return self.ast.nodes

    def get_nodes_types(self):
        return self.ast.get_nodes_type()

    @staticmethod
    def _normalize_ast(function):
        nodes = function.get("nodes") or []
        edges = function.get("edges") or []

        edges_by_src = {}
        for idx, edge in enumerate(edges):
            src = str(edge.get("src", ""))
            dst = str(edge.get("dst", ""))
            if not src or not dst:
                continue
            edge_type = edge.get("edgeType", "EDGE")
            edge_label = "Ast" if str(edge_type).upper() == "AST" else edge_type
            edge_id = f"{edge_label}@{idx}"
            edges_by_src.setdefault(src, []).append(
                {
                    "id": edge_id,
                    "in": src,
                    "out": dst,
                }
            )

        normalized_nodes = []
        for node in nodes:
            node_id = str(node.get("id", ""))
            properties = Function._normalize_properties(node.get("properties"))
            normalized_nodes.append(
                {
                    "id": node_id,
                    "properties": properties,
                    "edges": edges_by_src.get(node_id, []),
                }
            )

        return normalized_nodes

    @staticmethod
    def _normalize_properties(props):
        if props is None:
            return []
        if isinstance(props, dict):
            return [
                {"key": str(key), "value": "" if value is None else str(value)}
                for key, value in props.items()
            ]
        return props
