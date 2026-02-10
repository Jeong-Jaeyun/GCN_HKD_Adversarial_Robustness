
import re
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class TransformationResult:

    transformed_code: str
    transformation_type: str
    changes_made: Dict
    graph_changes: Optional[Dict] = None


class AdversarialTransformer:

    def __init__(
        self,
        variable_rename_prob: float = 0.3,
        dead_code_ratio: float = 0.2,
        control_flow_complexity: float = 0.4,
        seed: Optional[int] = None
    ):
        if not 0.0 <= variable_rename_prob <= 1.0:
            raise ValueError("variable_rename_prob must be in [0, 1]")
        if not 0.0 <= dead_code_ratio <= 1.0:
            raise ValueError("dead_code_ratio must be in [0, 1]")
        if not 0.0 <= control_flow_complexity <= 1.0:
            raise ValueError("control_flow_complexity must be in [0, 1]")

        self.variable_rename_prob = variable_rename_prob
        self.dead_code_ratio = dead_code_ratio
        self.control_flow_complexity = control_flow_complexity

        self._rng = random.Random(seed)
        self.variable_mapping: Dict[str, str] = {}

        self._transformation_aliases = {
            'variable_renaming': 'variable_renaming',
            'dead_code': 'dead_code',
            'dead_code_insertion': 'dead_code',
            'control_flow': 'control_flow',
            'control_flow_flattening': 'control_flow',
            'all': 'all'
        }

        self._keywords = {
            'if', 'else', 'while', 'for', 'return', 'int', 'float', 'double',
            'char', 'void', 'struct', 'class', 'def', 'main', 'include',
            'import', 'from', 'const', 'static', 'unsigned', 'signed', 'bool',
            'switch', 'case', 'break', 'continue', 'do', 'goto', 'enum',
            'sizeof', 'typedef', 'volatile', 'extern', 'register', 'union',
            'auto', 'long', 'short'
        }

    def transform(
        self,
        code: str,
        transformation_type: str = 'all'
    ) -> TransformationResult:
        canonical_type = self._normalize_transformation_type(transformation_type)
        transformed_code = code
        changes = {}
        self.variable_mapping = {}

        if canonical_type in ['variable_renaming', 'all']:
            transformed_code, var_changes = self._apply_variable_renaming(
                transformed_code
            )
            changes['variable_renaming'] = var_changes

        if canonical_type in ['dead_code', 'all']:
            transformed_code, dead_code_info = self._apply_dead_code_insertion(
                transformed_code
            )
            changes['dead_code_insertion'] = dead_code_info

        if canonical_type in ['control_flow', 'all']:
            transformed_code, cf_changes = self._apply_control_flow_flattening(
                transformed_code
            )
            changes['control_flow_flattening'] = cf_changes

        return TransformationResult(
            transformed_code=transformed_code,
            transformation_type=canonical_type,
            changes_made=changes
        )

    def _normalize_transformation_type(self, transformation_type: str) -> str:
        if transformation_type not in self._transformation_aliases:
            supported = sorted(self._transformation_aliases.keys())
            raise ValueError(
                f"Unknown transformation type: {transformation_type}. Supported: {supported}"
            )
        return self._transformation_aliases[transformation_type]

    def _apply_variable_renaming(self, code: str) -> Tuple[str, Dict]:
        variables = sorted(self._extract_variable_candidates(code))


        renaming_map: Dict[str, str] = {}
        transformed_code = code

        for var in variables:
            if self._rng.random() < self.variable_rename_prob:
                new_name = self._generate_unique_var_name(renaming_map.values())
                renaming_map[var] = new_name
                transformed_code = self._replace_identifier_safely(
                    transformed_code,
                    old_name=var,
                    new_name=new_name
                )

        self.variable_mapping = renaming_map
        return transformed_code, {
            'renaming_map': renaming_map,
            'num_renamed': len(renaming_map)
        }

    def _apply_dead_code_insertion(self, code: str) -> Tuple[str, Dict]:
        lines = code.split('\n')

        if not lines:
            return code, {'num_insertions': 0, 'insertion_points': [], 'dead_code_blocks': 0}

        dead_code_blocks = [
            "if (0) { int __dead_flag = 0; __dead_flag += 1; }",
            "{ volatile int __dead_local = 0; (void)__dead_local; }",
            "for (int __dead_i = 0; __dead_i < 0; ++__dead_i) { }",
        ]

        candidate_points = [
            idx for idx, line in enumerate(lines)
            if line.strip().endswith(';') or line.strip().endswith('{')
        ]
        if not candidate_points:
            candidate_points = list(range(len(lines)))

        num_insertions = int(len(lines) * self.dead_code_ratio)
        if self.dead_code_ratio > 0 and num_insertions == 0:
            num_insertions = 1
        num_insertions = min(num_insertions, len(candidate_points))
        insertion_points = self._rng.sample(candidate_points, num_insertions) if num_insertions > 0 else []

        inserted_blocks = []

        for idx in sorted(insertion_points, reverse=True):
            dead_block = self._rng.choice(dead_code_blocks)
            indent = re.match(r'^\s*', lines[idx]).group(0) if idx < len(lines) else ''
            lines.insert(idx + 1, f"{indent}{dead_block}")
            inserted_blocks.append(dead_block)

        transformed_code = '\n'.join(lines)

        return transformed_code, {
            'num_insertions': len(insertion_points),
            'insertion_points': sorted(insertion_points),
            'dead_code_blocks': inserted_blocks
        }

    def _apply_control_flow_flattening(self, code: str) -> Tuple[str, Dict]:


        lines = code.split('\n')
        transformed_lines = []


        cf_changes = []

        for idx, line in enumerate(lines):
            transformed_lines.append(line)


            if any(kw in line for kw in ['if', 'while', 'for']) and self._rng.random() < self.control_flow_complexity:
                indent = re.match(r'^\s*', line).group(0)
                block = f"{indent}if (0) {{ int __cf_dummy_{idx} = 0; __cf_dummy_{idx}++; }}"
                transformed_lines.append(block)
                cf_changes.append({
                    'line': idx,
                    'type': 'unreachable_branch_added'
                })

        transformed_code = '\n'.join(transformed_lines)

        return transformed_code, {
            'control_flow_changes': cf_changes,
            'num_flattening_ops': len(cf_changes)
        }

    def _extract_variable_candidates(self, code: str) -> Set[str]:
        declaration_pattern = re.compile(
            r'\b(?:int|float|double|char|long|short|unsigned|signed|size_t|bool|auto)\s+([A-Za-z_][A-Za-z0-9_]*)\b'
        )
        assignment_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=')

        declarations = set(declaration_pattern.findall(code))
        assignments = set(assignment_pattern.findall(code))
        raw_candidates = declarations | assignments

        candidates = {
            token for token in raw_candidates
            if token not in self._keywords
            and not token.startswith('__')
            and token not in {'printf', 'scanf', 'malloc', 'free'}
        }
        return candidates

    def _generate_unique_var_name(self, used_names) -> str:
        used_name_set = set(used_names)
        while True:
            candidate = f"var_{self._rng.randint(10000, 99999)}"
            if candidate not in used_name_set and candidate not in self._keywords:
                return candidate

    def _replace_identifier_safely(self, code: str, old_name: str, new_name: str) -> str:
        if old_name == new_name:
            return code

        identifier_pattern = re.compile(rf'\b{re.escape(old_name)}\b')
        literal_or_comment_re = re.compile(
            r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|//.*?$|/\*.*?\*/)",
            re.MULTILINE | re.DOTALL
        )

        parts = literal_or_comment_re.split(code)
        for idx, part in enumerate(parts):

            if idx % 2 == 1:
                continue

            safe_lines = []
            for line in part.splitlines(keepends=True):
                if line.lstrip().startswith('#'):
                    safe_lines.append(line)
                else:
                    safe_lines.append(identifier_pattern.sub(new_name, line))
            parts[idx] = ''.join(safe_lines)

        return ''.join(parts)


class AdversarialPerturbationGenerator:

    def __init__(
        self,
        num_samples: int = 5,
        transformation_types: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.transformation_types = transformation_types or [
            'variable_renaming',
            'dead_code',
            'control_flow',
            'all'
        ]

        self._rng = random.Random(seed)
        self.transformer = AdversarialTransformer(seed=seed)

    def generate(self, code: str) -> List[TransformationResult]:
        perturbations = []

        for i in range(self.num_samples):

            transform_type = self._rng.choice(self.transformation_types)


            result = self.transformer.transform(code, transform_type)
            perturbations.append(result)

        return perturbations
