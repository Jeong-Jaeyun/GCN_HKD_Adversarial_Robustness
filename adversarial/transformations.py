"""Adversarial code transformations (Phase 1).

Applies semantic-preserving transformations that modify code structure:
- Variable renaming
- Dead code insertion  
- Control flow flattening

These transformations change the graph structure while preserving semantics,
allowing us to test adversarial robustness.
"""

import re
import random
from typing import Tuple, Dict, List, Optional
import networkx as nx
from dataclasses import dataclass


@dataclass
class TransformationResult:
    """Result of applying a transformation."""
    
    transformed_code: str
    transformation_type: str
    changes_made: Dict
    graph_changes: Optional[Dict] = None


class AdversarialTransformer:
    """Applies adversarial transformations to code while preserving semantics.
    
    Implements Phase 1 of the architecture: semantic-preserving graph mutations.
    """
    
    def __init__(
        self,
        variable_rename_prob: float = 0.3,
        dead_code_ratio: float = 0.2,
        control_flow_complexity: float = 0.4,
        seed: Optional[int] = None
    ):
        """Initialize adversarial transformer.
        
        Args:
            variable_rename_prob: Probability of renaming each variable
            dead_code_ratio: Ratio of dead code to insert
            control_flow_complexity: Factor controlling control flow flattening
            seed: Random seed
        """
        self.variable_rename_prob = variable_rename_prob
        self.dead_code_ratio = dead_code_ratio
        self.control_flow_complexity = control_flow_complexity
        
        if seed is not None:
            random.seed(seed)
        
        self.variable_mapping: Dict[str, str] = {}
    
    def transform(
        self,
        code: str,
        transformation_type: str = 'all'
    ) -> TransformationResult:
        """Apply adversarial transformation to code.
        
        Args:
            code: Source code string
            transformation_type: Which transformation to apply
                - 'variable_renaming': Rename variables
                - 'dead_code': Insert dead code
                - 'control_flow': Flatten control flow
                - 'all': Apply all transformations
                
        Returns:
            TransformationResult with transformed code and changes
        """
        transformed_code = code
        changes = {}
        
        if transformation_type in ['variable_renaming', 'all']:
            transformed_code, var_changes = self._apply_variable_renaming(
                transformed_code
            )
            changes['variable_renaming'] = var_changes
        
        if transformation_type in ['dead_code', 'all']:
            transformed_code, dead_code_info = self._apply_dead_code_insertion(
                transformed_code
            )
            changes['dead_code_insertion'] = dead_code_info
        
        if transformation_type in ['control_flow', 'all']:
            transformed_code, cf_changes = self._apply_control_flow_flattening(
                transformed_code
            )
            changes['control_flow_flattening'] = cf_changes
        
        return TransformationResult(
            transformed_code=transformed_code,
            transformation_type=transformation_type,
            changes_made=changes
        )
    
    def _apply_variable_renaming(self, code: str) -> Tuple[str, Dict]:
        """Rename variables with random names.
        
        Changes variable names while preserving semantics.
        Complicates graph structure to test robustness.
        
        Returns:
            Tuple of (transformed_code, renaming_mapping)
        """
        # Extract variable names (simple regex-based approach)
        # In production, use proper AST parsing
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        # Find all variables
        variables = set(re.findall(var_pattern, code))
        
        # Filter out keywords
        keywords = {
            'if', 'else', 'while', 'for', 'return', 'int', 'float', 'char',
            'void', 'struct', 'class', 'def', 'main', 'include', 'import',
            'from', 'const', 'static', 'unsigned', 'signed', 'bool'
        }
        variables = variables - keywords
        
        # Rename variables with probability
        renaming_map = {}
        transformed_code = code
        
        for var in variables:
            if random.random() < self.variable_rename_prob:
                # Generate random new name
                new_name = f"var_{random.randint(10000, 99999)}"
                renaming_map[var] = new_name
                
                # Replace all occurrences (simple word-boundary replacement)
                pattern = r'\b' + re.escape(var) + r'\b'
                transformed_code = re.sub(pattern, new_name, transformed_code)
        
        return transformed_code, {'renaming_map': renaming_map, 'num_renamed': len(renaming_map)}
    
    def _apply_dead_code_insertion(self, code: str) -> Tuple[str, Dict]:
        """Insert dead code that doesn't affect semantics.
        
        Adds non-functional code blocks to increase noise in the graph.
        Useful for testing robustness to irrelevant graph structures.
        
        Returns:
            Tuple of (transformed_code, insertion_info)
        """
        lines = code.split('\n')
        
        # Identify insertion points (after each function)
        dead_code_blocks = [
            "int _dead_var_1 = 0;",
            "if (0) { printf(\"dead code\"); }",
            "{ int x = 5; x = x + 1; }",
            "void _dead_func() { }",
            "// This is dead code\nint unused_var = 0;",
        ]
        
        num_insertions = max(1, int(len(lines) * self.dead_code_ratio))
        insertion_points = random.sample(
            range(len(lines)),
            min(num_insertions, len(lines))
        )
        
        # Insert dead code (in reverse order to preserve line numbers)
        for idx in sorted(insertion_points, reverse=True):
            dead_block = random.choice(dead_code_blocks)
            lines.insert(idx, "\n" + dead_block + "\n")
        
        transformed_code = '\n'.join(lines)
        
        return transformed_code, {
            'num_insertions': len(insertion_points),
            'insertion_points': insertion_points,
            'dead_code_blocks': num_insertions
        }
    
    def _apply_control_flow_flattening(self, code: str) -> Tuple[str, Dict]:
        """Flatten control flow structures.
        
        Converts nested if/else and loops into flattened structures,
        making the CFG more complex while preserving behavior.
        
        Returns:
            Tuple of (transformed_code, transformation_info)
        """
        # This is a simplified version
        # Real CFG flattening requires proper AST transformation
        
        lines = code.split('\n')
        transformed_lines = []
        
        # Insert additional control flow edges
        cf_changes = []
        
        for idx, line in enumerate(lines):
            transformed_lines.append(line)
            
            # After conditional statements, add unreachable branches
            if any(kw in line for kw in ['if', 'while', 'for']):
                # Add dummy label (simulating flattened CFG)
                label_name = f"__label_block_{idx}:"
                
                if random.random() < self.control_flow_complexity:
                    transformed_lines.append(f"// {label_name}")
                    cf_changes.append({'line': idx, 'type': 'label_added'})
        
        transformed_code = '\n'.join(transformed_lines)
        
        return transformed_code, {
            'control_flow_changes': cf_changes,
            'num_flattening_ops': len(cf_changes)
        }


class AdversarialPerturbationGenerator:
    """Generates multiple adversarial examples from a single code sample."""
    
    def __init__(
        self,
        num_samples: int = 5,
        transformation_types: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """Initialize perturbation generator.
        
        Args:
            num_samples: Number of perturbations per sample
            transformation_types: Types of transformations to apply
            seed: Random seed
        """
        self.num_samples = num_samples
        self.transformation_types = transformation_types or [
            'variable_renaming',
            'dead_code',
            'control_flow',
            'all'
        ]
        
        self.transformer = AdversarialTransformer(seed=seed)
    
    def generate(self, code: str) -> List[TransformationResult]:
        """Generate multiple adversarial perturbations.
        
        Args:
            code: Original source code
            
        Returns:
            List of TransformationResult objects
        """
        perturbations = []
        
        for i in range(self.num_samples):
            # Randomly select transformation type
            transform_type = random.choice(self.transformation_types)
            
            # Apply transformation
            result = self.transformer.transform(code, transform_type)
            perturbations.append(result)
        
        return perturbations
