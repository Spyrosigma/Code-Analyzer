import os
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Set, Tuple
import json

class CodeMetrics:
    def __init__(self):
        self.node_frequencies = defaultdict(int)
        self.function_calls = defaultdict(set)
        self.class_instances = defaultdict(set)
        self.import_usage = defaultdict(set)
        self.method_calls = defaultdict(lambda: defaultdict(set))
        
    def update_frequency(self, node_type: str):
        self.node_frequencies[node_type] += 1

class CodeAnalyzer:
    def __init__(self):

        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)

        # Graph for storing relationships
        self.dependency_graph = nx.DiGraph()
        self.metrics = CodeMetrics()
        # Storage for analyzed data
        self.functions = defaultdict(dict)
        self.classes = defaultdict(dict)
        self.imports = defaultdict(set)
        
    # def analyze_file(self, file_path: str) -> None:
    #     """Analyze a single file and extract its components."""
    #     with open(file_path, 'rb') as f:
    #         content = f.read()
            
    #     tree = self.parser.parse(content)
        
    #     # Extract components
    #     self._extract_functions(tree.root_node, file_path)
    #     self._extract_classes(tree.root_node, file_path)
    #     self._extract_imports(tree.root_node, file_path)

    def analyze_file(self, file_path: str) -> None:
        """Analyze a single file and extract its components."""
        with open(file_path, 'rb') as f:
            content = f.read()
            
        tree = self.parser.parse(content)
        
        # Extract basic components
        self._extract_functions(tree.root_node, file_path)
        self._extract_classes(tree.root_node, file_path)
        self._extract_imports(tree.root_node, file_path)
    
        # Perform enhanced relationship analysis
        self._analyze_relationships(tree.root_node)
        
    def _extract_functions(self, node, file_path: str) -> None:
        """Extract function definitions and their relationships."""
        if node.type == 'function_definition':
            name_node = next(
                (child for child in node.children if child.type == 'identifier'), 
                None
            )
            if name_node:
                func_name = name_node.text.decode('utf-8')
                self.functions[file_path][func_name] = {
                    'start': node.start_point,
                    'end': node.end_point,
                    'calls': self._extract_function_calls(node),
                    'context': self._extract_context(node)
                }
                
        for child in node.children:
            self._extract_functions(child, file_path)
            
    def _extract_function_calls(self, node) -> Set[str]:
        """Extract all function calls within a node."""
        calls = set()
        
        def visit_node(node):
            if node.type == 'call':
                func_node = node.children[0]
                if func_node.type == 'identifier':
                    calls.add(func_node.text.decode('utf-8'))
            for child in node.children:
                visit_node(child)
                
        visit_node(node)
        return calls
    
    def _extract_context(self, node) -> Dict:
        """Extract contextual information around the node."""
        context = {
            'docstring': None,
            'decorators': [],
            'parameters': []
        }
        
        # Extract docstring
        for child in node.children:
            if child.type == 'expression_statement':
                string_node = child.children[0]
                if string_node.type in ('string', 'string_content'):
                    context['docstring'] = string_node.text.decode('utf-8')
                    break
                    
        # Extract parameters
        param_list = next(
            (child for child in node.children if child.type == 'parameters'), 
            None
        )
        if param_list:
            for param in param_list.children:
                if param.type == 'identifier':
                    context['parameters'].append(param.text.decode('utf-8'))
                    
        return context
    
    def _extract_classes(self, node, file_path: str) -> None:
        """Extract class definitions and their relationships."""
        if node.type == 'class_definition':
            name_node = next(
                (child for child in node.children if child.type == 'identifier'), 
                None
            )
            if name_node:
                class_name = name_node.text.decode('utf-8')
                self.classes[file_path][class_name] = {
                    'methods': self._extract_methods(node),
                    'inheritance': self._extract_inheritance(node)
                }
                
        for child in node.children:
            self._extract_classes(child, file_path)
            
    def _extract_methods(self, class_node) -> Dict[str, Dict]:
        """Extract methods from a class definition."""
        methods = {}
        for child in class_node.children:
            if child.type == 'function_definition':
                name_node = next(
                    (c for c in child.children if c.type == 'identifier'), 
                    None
                )
                if name_node:
                    method_name = name_node.text.decode('utf-8')
                    methods[method_name] = {
                        'calls': self._extract_function_calls(child),
                        'context': self._extract_context(child)
                    }
        return methods
    
    def _extract_inheritance(self, class_node) -> List[str]:
        """Extract inheritance information from a class definition."""
        inheritance = []
        for child in class_node.children:
            if child.type == 'argument_list':
                for arg in child.children:
                    if arg.type == 'identifier':
                        inheritance.append(arg.text.decode('utf-8'))
        return inheritance
    
    def _extract_imports(self, node, file_path: str) -> None:
        """Extract import statements and their relationships."""
        if node.type in ('import_statement', 'import_from_statement'):
            module_name = None
            for child in node.children:
                if child.type == 'dotted_name':
                    module_name = child.text.decode('utf-8')
                    self.imports[file_path].add(module_name)
                    
        for child in node.children:
            self._extract_imports(child, file_path)
            
    def create_chunks(self, max_chunk_size: int = 1000) -> List[Dict]:
        """Create semantic chunks of the codebase."""
        chunks = []
        current_chunk = {
            'content': [],
            'context': [],
            'relationships': []
        }
        
        # Group related functions and classes
        for file_path, file_functions in self.functions.items():
            for func_name, func_data in file_functions.items():
                # Create chunk based on function and its related calls
                chunk_content = {
                    'type': 'function',
                    'name': func_name,
                    'file': file_path,
                    'content': {
                        'start': func_data['start'],
                        'end': func_data['end'],
                        'calls': list(func_data['calls']),  # Convert set to list
                        'context': func_data['context']
                    }
                }
                
                # Add relationships
                relationships = []
                for call in func_data['calls']:
                    relationships.append({
                        'type': 'calls',
                        'target': call
                    })
                
                if len(str(current_chunk)) + len(str(chunk_content)) > max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = {
                        'content': [],
                        'context': [],
                        'relationships': []
                    }
                    
                current_chunk['content'].append(chunk_content)
                current_chunk['relationships'].extend(relationships)
                
        if current_chunk['content']:
            chunks.append(current_chunk)
            
        return chunks
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive analysis report."""
        return {
            'files_analyzed': len(self.functions),
            'total_functions': sum(len(funcs) for funcs in self.functions.values()),
            'total_classes': sum(len(classes) for classes in self.classes.values()),
            'imports': {
                file: list(imports)  # Convert set to list
                for file, imports in self.imports.items()
            },
            'function_relationships': {
                file: {
                    func: list(data['calls'])  # Convert set to list
                    for func, data in file_funcs.items()
                }
                for file, file_funcs in self.functions.items()
            },
            'class_hierarchy': {
                file: {
                    class_name: data['inheritance']
                    for class_name, data in file_classes.items()
                }
                for file, file_classes in self.classes.items()
            }
        }

    def _analyze_relationships(self, node, current_function=None, current_class=None):
        """Analyze deeper relationships between code elements."""
        if node is None:
            return
            
        # Track node frequencies
        self.metrics.update_frequency(node.type)
        
        # Analyze function calls and their context
        if node.type == 'call':
            func_node = next((child for child in node.children if child.type == 'identifier'), None)
            if func_node:
                func_name = func_node.text.decode('utf-8')
                if current_function:
                    self.metrics.function_calls[current_function].add(func_name)
                if current_class:
                    self.metrics.method_calls[current_class][current_function or 'unknown'].add(func_name)

        # Track class instantiations
        if node.type == 'class_definition':
            class_name = next(
                (child.text.decode('utf-8') 
                for child in node.children if child.type == 'identifier'),
                None
            )
            if class_name:
                self._analyze_class_usage(node, class_name)

        # Analyze import usage
        if node.type in ('import_statement', 'import_from_statement'):
            self._track_import_usage(node, current_function, current_class)

        # Recursive analysis
        for child in node.children:
            self._analyze_relationships(
                child,
                current_function=current_function,
                current_class=current_class
            )

    def _analyze_class_usage(self, node, class_name: str):
        """Analyze how classes are used throughout the code."""
        def visit_node(n):
            if n.type == 'call':
                caller = next(
                    (child.text.decode('utf-8') 
                    for child in n.children if child.type == 'identifier'),
                    None
                )
                if caller == class_name:
                    self.metrics.class_instances[class_name].add(
                        f"{node.start_point[0]}:{node.start_point[1]}"
                    )
            for child in n.children:
                visit_node(child)
        
        visit_node(node)

    def _track_import_usage(self, node, current_function: str, current_class: str):
        """Track where imports are used in the code."""
        import_name = next(
            (child.text.decode('utf-8') 
            for child in node.children if child.type == 'dotted_name'),
            None
        )
        if import_name:
            context = f"{current_class or ''}.{current_function or ''}"
            self.metrics.import_usage[import_name].add(context.strip('.'))

    def enhanced_chunk_creation(self) -> List[Dict]:
        """Create more sophisticated chunks based on code analysis."""
        chunks = []
        
        # Group by logical units (classes and their related functions)
        for file_path, classes in self.classes.items():
            for class_name, class_data in classes.items():
                class_chunk = {
                    'type': 'class',
                    'name': class_name,
                    'file': file_path,
                    'methods': class_data['methods'],
                    'inheritance': class_data['inheritance'],
                    'instances': list(self.metrics.class_instances[class_name]),
                    'method_relationships': {
                        method: list(calls)
                        for method, calls in self.metrics.method_calls[class_name].items()
                    }
                }
                chunks.append(class_chunk)
        
        # Group related functions (based on call patterns)
        function_groups = self._group_related_functions()
        for group in function_groups:
            func_chunk = {
                'type': 'function_group',
                'functions': [
                    {
                        'name': func,
                        'calls': list(self.metrics.function_calls[func]),
                        'import_dependencies': [
                            imp for imp, usages in self.metrics.import_usage.items()
                            if func in usages
                        ]
                    }
                    for func in group
                ]
            }
            chunks.append(func_chunk)
        
        return chunks

    def _group_related_functions(self) -> List[Set[str]]:
        """Group functions based on their relationships."""
        groups = []
        processed = set()
        
        for func in self.metrics.function_calls:
            if func in processed:
                continue
                
            # Find related functions through call graph
            related = {func}
            to_process = {func}
            
            while to_process:
                current = to_process.pop()
                calls = self.metrics.function_calls[current]
                for called_func in calls:
                    if called_func not in processed:
                        related.add(called_func)
                        to_process.add(called_func)
            
            processed.update(related)
            groups.append(related)
        
        return groups

    def generate_enhanced_report(self) -> Dict:
        """Generate a more detailed analysis report."""
        base_report = self.generate_report()
        
        enhanced_report = {
            **base_report,
            'metrics': {
                'node_frequencies': dict(self.metrics.node_frequencies),
                'function_call_patterns': {
                    func: list(calls)
                    for func, calls in self.metrics.function_calls.items()
                },
                'class_usage': {
                    class_name: list(instances)
                    for class_name, instances in self.metrics.class_instances.items()
                },
                'import_dependencies': {
                    import_name: list(usages)
                    for import_name, usages in self.metrics.import_usage.items()
                }
            },
            'chunks': self.enhanced_chunk_creation()
        }
        
        return enhanced_report

# Example usage
# def analyze_codebase(root_dir: str) -> None:
#     analyzer = CodeAnalyzer()
    
#     # Analyze all Python files
#     for root, _, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith('.py'):
#                 file_path = os.path.join(root, file)
#                 analyzer.analyze_file(file_path)
    
#     # Generate chunks and report
#     chunks = analyzer.create_chunks()
#     # print('CHUNKS---------', chunks)
#     report = analyzer.generate_report()
#     # print('Report ----------',report)

#     # Save results
#     with open('analysis_report.json', 'w') as f:
#         json.dump(report, f, indent=4)
        
#     with open('code_chunks.json', 'w') as f:
#         json.dump(chunks, f, indent=4)

def analyze_codebase(root_dir: str, ignore_dirs: set = None) -> None:
    """
    Analyze Python files in the codebase while ignoring specified directories.
    
    Args:
        root_dir (str): Root directory of the codebase
        ignore_dirs (set): Set of directory names to ignore (default: common env directories)
    """
    if ignore_dirs is None:
        ignore_dirs = {
            'venv',          # virtual environment
            'env',           # another common venv name
            '.env',          # hidden venv
            '__pycache__',   # Python cache
            'node_modules',  # Node.js modules
            '.git',          # Git directory
            'build',         # Build directory
            'dist',          # Distribution directory
            'site-packages', # Python packages
            '.pytest_cache', # Pytest cache
            '.mypy_cache',   # MyPy cache
            '.tox'          # Tox testing
        }
    
    analyzer = CodeAnalyzer()
    
    # Analyze all Python files
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Analyzing: {file_path}")  # Optional: for visibility
                analyzer.analyze_file(file_path)
    
    # Generate enhanced report and chunks
    enhanced_report = analyzer.generate_enhanced_report()
    chunks = analyzer.enhanced_chunk_creation()

    # Save results
    with open('analysis_report.json', 'w') as f:
        json.dump(enhanced_report, f, indent=4)
        
    with open('code_chunks.json', 'w') as f:
        json.dump(chunks, f, indent=4)

    # Print summary with cleaner output
    print("\nAnalysis complete!")
    print(f"Found {enhanced_report['total_functions']} functions")
    print(f"Found {enhanced_report['total_classes']} classes")
    print(f"Generated {len(chunks)} code chunks")
    print("\nIgnored directories:", ', '.join(sorted(ignore_dirs)))

# Example usage:
if __name__ == "__main__":
    # Default usage with predefined ignore list
    analyze_codebase("./")
    
    # Or with custom ignore list
    # custom_ignore = {'venv', 'tests', 'docs'}
    # analyze_codebase("./StreetSpecter", ignore_dirs=custom_ignore)