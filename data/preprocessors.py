"""Graph construction for CFG and DFG from source and binary code."""

import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class GraphNode:
    """Represents a node in the code graph."""
    
    idx: int
    node_type: str  # 'instruction', 'basic_block', 'function', etc.
    content: str
    features: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash((self.idx, self.node_type))


@dataclass
class GraphEdge:
    """Represents an edge in the code graph."""
    
    source_idx: int
    target_idx: int
    edge_type: str  # 'control_flow', 'data_flow', 'call_graph', etc.
    weight: float = 1.0


class BaseGraphConstructor(ABC):
    """Abstract base class for graph construction."""
    
    def __init__(self, max_nodes: int = 10000, feature_dim: int = 768):
        """Initialize graph constructor.
        
        Args:
            max_nodes: Maximum number of nodes to include in graph
            feature_dim: Feature dimension for node representations
        """
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
    
    @abstractmethod
    def construct_from_source(self, source_code: str) -> nx.DiGraph:
        """Construct graph from source code."""
        pass
    
    @abstractmethod
    def construct_from_binary(self, binary_path: str) -> nx.DiGraph:
        """Construct graph from binary."""
        pass
    
    def _create_networkx_graph(self) -> nx.DiGraph:
        """Convert internal node/edge representation to NetworkX graph."""
        G = nx.DiGraph()
        
        # Add nodes with features
        for node_idx, node in self.nodes.items():
            G.add_node(
                node_idx,
                type=node.node_type,
                content=node.content,
                features=node.features
            )
        
        # Add edges with types and weights
        for edge in self.edges:
            G.add_edge(
                edge.source_idx,
                edge.target_idx,
                type=edge.edge_type,
                weight=edge.weight
            )
        
        return G
    
    def get_graph(self) -> nx.DiGraph:
        """Get the constructed graph as NetworkX object."""
        return self._create_networkx_graph()


class CFGConstructor(BaseGraphConstructor):
    """Control Flow Graph constructor.
    
    Constructs CFG from intermediate representation (IR) or binary code.
    Nodes represent basic blocks, edges represent control flow.
    """
    
    def __init__(self, max_nodes: int = 10000, feature_dim: int = 768):
        super().__init__(max_nodes, feature_dim)
    
    def construct_from_source(self, source_code: str) -> nx.DiGraph:
        """Construct CFG from source code.
        
        Args:
            source_code: Source code string
            
        Returns:
            NetworkX directed graph representing CFG
        """
        # Parse source code to extract control flow
        # This would typically use AST parsing or intermediate representation
        self.nodes = self._extract_basic_blocks(source_code)
        self.edges = self._build_control_flow_edges(source_code)
        
        return self.get_graph()
    
    def construct_from_binary(self, binary_path: str) -> nx.DiGraph:
        """Construct CFG from binary executable.
        
        Args:
            binary_path: Path to binary file
            
        Returns:
            NetworkX directed graph representing CFG
        """
        # Extract CFG from binary using disassembly
        # Typically uses tools like IDA, Ghidra, or Radare2
        self.nodes = self._extract_basic_blocks_from_binary(binary_path)
        self.edges = self._build_control_flow_edges_from_binary(binary_path)
        
        return self.get_graph()
    
    def _extract_basic_blocks(self, source_code: str) -> Dict[int, GraphNode]:
        """Extract basic blocks from source code."""
        blocks = {}
        # Simple tokenization (in practice, use proper AST/IR parsing)
        lines = source_code.split('\n')
        block_idx = 0
        current_block_content = []
        
        for line_idx, line in enumerate(lines):
            if any(kw in line for kw in ['if', 'while', 'for', 'switch', 'else']):
                if current_block_content:
                    blocks[block_idx] = GraphNode(
                        idx=block_idx,
                        node_type='basic_block',
                        content='\n'.join(current_block_content)
                    )
                    block_idx += 1
                    current_block_content = []
            current_block_content.append(line)
        
        if current_block_content:
            blocks[block_idx] = GraphNode(
                idx=block_idx,
                node_type='basic_block',
                content='\n'.join(current_block_content)
            )
        
        return blocks
    
    def _extract_basic_blocks_from_binary(self, binary_path: str) -> Dict[int, GraphNode]:
        """Extract basic blocks from binary."""
        # Placeholder implementation
        # In practice, use binary analysis tools
        return {}
    
    def _build_control_flow_edges(self, source_code: str) -> List[GraphEdge]:
        """Build control flow edges from source code."""
        edges = []
        lines = source_code.split('\n')
        
        for idx, line in enumerate(lines):
            next_idx = idx + 1
            # Sequential control flow
            edges.append(GraphEdge(
                source_idx=idx,
                target_idx=next_idx,
                edge_type='sequential'
            ))
            
            # Conditional/loop edges
            if any(kw in line for kw in ['if', 'while', 'for']):
                # Find corresponding else/end blocks
                edges.append(GraphEdge(
                    source_idx=idx,
                    target_idx=next_idx + 5,  # Simplified
                    edge_type='conditional'
                ))
        
        return edges
    
    def _build_control_flow_edges_from_binary(self, binary_path: str) -> List[GraphEdge]:
        """Build control flow edges from binary."""
        # Placeholder implementation
        return []


class DFGConstructor(BaseGraphConstructor):
    """Data Flow Graph constructor.
    
    Constructs DFG from source or binary code.
    Nodes represent variables/values, edges represent data dependencies.
    """
    
    def __init__(self, max_nodes: int = 10000, feature_dim: int = 768):
        super().__init__(max_nodes, feature_dim)
    
    def construct_from_source(self, source_code: str) -> nx.DiGraph:
        """Construct DFG from source code."""
        self.nodes = self._extract_variables(source_code)
        self.edges = self._build_data_flow_edges(source_code)
        
        return self.get_graph()
    
    def construct_from_binary(self, binary_path: str) -> nx.DiGraph:
        """Construct DFG from binary."""
        self.nodes = self._extract_variables_from_binary(binary_path)
        self.edges = self._build_data_flow_edges_from_binary(binary_path)
        
        return self.get_graph()
    
    def _extract_variables(self, source_code: str) -> Dict[int, GraphNode]:
        """Extract variables and their definitions."""
        variables = {}
        # Simple variable extraction (use proper SSA/IR in practice)
        import re
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = re.finditer(var_pattern, source_code)
        
        for idx, match in enumerate(matches):
            var_name = match.group()
            variables[idx] = GraphNode(
                idx=idx,
                node_type='variable',
                content=var_name
            )
            if idx >= self.max_nodes:
                break
        
        return variables
    
    def _extract_variables_from_binary(self, binary_path: str) -> Dict[int, GraphNode]:
        """Extract register/memory locations from binary."""
        # Placeholder implementation
        return {}
    
    def _build_data_flow_edges(self, source_code: str) -> List[GraphEdge]:
        """Build data flow edges from assignments and uses."""
        edges = []
        lines = source_code.split('\n')
        
        for line_idx, line in enumerate(lines):
            if '=' in line and not '==' in line:
                # Assignment: extract lhs (definition) and rhs (uses)
                parts = line.split('=')
                lhs = parts[0].strip()
                rhs = parts[1].strip() if len(parts) > 1 else ''
                
                # Simple token-based data flow
                for var_idx, node in enumerate(self.nodes.items()):
                    if lhs in node[1].content:
                        for var_idx2, node2 in enumerate(self.nodes.items()):
                            if rhs and node2[1].content in rhs:
                                edges.append(GraphEdge(
                                    source_idx=var_idx2,
                                    target_idx=var_idx,
                                    edge_type='data_flow'
                                ))
        
        return edges
    
    def _build_data_flow_edges_from_binary(self, binary_path: str) -> List[GraphEdge]:
        """Build data flow edges from binary analysis."""
        # Placeholder implementation
        return []


class HybridGraphConstructor(BaseGraphConstructor):
    """Constructs combined CFG+DFG hybrid graph."""
    
    def __init__(self, max_nodes: int = 10000, feature_dim: int = 768):
        super().__init__(max_nodes, feature_dim)
        self.cfg_constructor = CFGConstructor(max_nodes, feature_dim)
        self.dfg_constructor = DFGConstructor(max_nodes, feature_dim)
    
    def construct_from_source(self, source_code: str) -> nx.DiGraph:
        """Construct hybrid CFG+DFG from source code."""
        cfg = self.cfg_constructor.construct_from_source(source_code)
        dfg = self.dfg_constructor.construct_from_source(source_code)
        
        return self._merge_graphs(cfg, dfg)
    
    def construct_from_binary(self, binary_path: str) -> nx.DiGraph:
        """Construct hybrid CFG+DFG from binary."""
        cfg = self.cfg_constructor.construct_from_binary(binary_path)
        dfg = self.dfg_constructor.construct_from_binary(binary_path)
        
        return self._merge_graphs(cfg, dfg)
    
    def _merge_graphs(self, cfg: nx.DiGraph, dfg: nx.DiGraph) -> nx.DiGraph:
        """Merge CFG and DFG into a single graph."""
        G = nx.DiGraph()
        
        # Add all CFG nodes and edges
        G.add_nodes_from(cfg.nodes(data=True))
        G.add_edges_from(cfg.edges(data=True))
        
        # Add all DFG nodes and edges
        G.add_nodes_from(dfg.nodes(data=True))
        G.add_edges_from(dfg.edges(data=True))
        
        return G
