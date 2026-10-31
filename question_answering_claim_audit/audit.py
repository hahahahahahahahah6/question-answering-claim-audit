import ast
import fnmatch
import os
import re
from pathlib import Path
from .phrases import POSITIVE, NEGATIVE


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append('')
        elif match:
            fence = match[1]
            lines.append('')
        else:
            lines.append(line)
    return '\n'.join(lines)


def consume(text, phrases):
    found = []
    for phrase in sorted(phrases, key=lambda p: (-len(p), p)):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text):
            found.append(phrase)
            text = re.sub(pattern, ' ', text)
    return text, found


def claims(text):
    positive, negative = [], []
    for paragraph in re.split(r'\n\s*\n', prose(text)):
        remainder, neg = consume(normalize(paragraph), NEGATIVE)
        _, pos = consume(remainder, POSITIVE)
        positive.extend(pos)
        negative.extend(neg)
    return positive, negative


READERS = {'FARMReader', 'TransformersReader', 'ExtractiveReader'}


def evidence(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = node.module + '.' + alias.name

    def key(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = key(node.value)
            return base + '.' + node.attr if base else None
        return None

    def resolve(node, blocked, providers):
        name = key(node)
        if not name or name.split('.')[0] in blocked:
            return ''
        head, *tail = name.split('.')
        return '.'.join([providers.get(head, ''), *tail]) if head in providers else ''

    def reader(node, blocked, providers):
        if not isinstance(node, ast.Call):
            return False
        name = resolve(node.func, blocked, providers)
        return name.startswith('haystack.') and name.split('.')[-1] in READERS

    def scan(scope, inherited, bindings, class_scope=False, providers=None):
        providers = dict(aliases if providers is None else providers)
        blocked = set(inherited)
        # Definitions and assignments in this lexical scope shadow imports.
        nodes = []
        def collect(node):
            nodes.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return
            for child in ast.iter_child_nodes(node):
                collect(child)
        for statement in scope:
            collect(statement)
        for node in nodes:
            imported = {}
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imported[alias.asname or alias.name] = node.module + '.' + alias.name
            providers.update(imported)
            blocked.difference_update(imported)
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                blocked.add(node.name)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                blocked.add(node.id)
        bound = set(bindings) - blocked
        if class_scope:
            for statement in scope:
                for assignment in ast.walk(statement):
                    if isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
                        for target in targets:
                            name = key(target)
                            if name and name.startswith('self.') and reader(assignment.value, blocked, providers):
                                bound.add(name)
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    name = key(target)
                    if name:
                        bound.discard(name)
                        if reader(node.value, blocked, providers):
                            bound.add(name)
            if isinstance(node, ast.Call):
                name = resolve(node.func, blocked, providers)
                leaf = name.split('.')[-1]
                if name.startswith('transformers.'):
                    if leaf == 'QuestionAnsweringPipeline':
                        return True
                    if leaf == 'pipeline':
                        args = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == 'task']
                        for value in args:
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                task = normalize(value.value.replace('_', ' '))
                                if re.search(r'(?<![a-z])question answering(?![a-z])', task):
                                    return True
                if key(node.func) in bound:
                    return True
                if isinstance(node.func, ast.Attribute):
                    if key(node.func.value) in bound or reader(node.func.value, blocked, providers):
                        return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                extra = set(blocked)
                if hasattr(node, 'args'):
                    extra.update(a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg))
                if scan(node.body, extra, bound, isinstance(node, ast.ClassDef), providers):
                    return True
        return False
    return scan(tree.body, set(), set())


def audit(root, ignores=()):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('root must be an existing directory')
    positives, negatives, production, tests = [], [], [], []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory)/d).is_symlink())
        for filename in sorted(files):
            path = Path(directory)/filename
            rel = path.relative_to(root)
            name = rel.as_posix()
            if path.is_symlink() or any(fnmatch.fnmatch(name, p) or any(fnmatch.fnmatch(part, p) for part in rel.parts) for p in ignores):
                continue
            is_doc = name == 'README.md' or ('docs' in rel.parts[:-1] and path.suffix == '.md')
            if is_doc:
                pos, neg = claims(path.read_text(encoding='utf-8'))
                positives.extend({'path': name, 'phrase': p} for p in pos)
                negatives.extend({'path': name, 'phrase': p} for p in neg)
            if path.suffix == '.py' and not {'fixtures', 'docs', 'examples'}.intersection(rel.parts[:-1]):
                if evidence(path.read_text(encoding='utf-8')):
                    test = 'tests' in rel.parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
                    (tests if test else production).append(name)
    findings = []
    if positives and not production:
        findings.append({'id': 'documented-question-answering-claim-false', 'severity': 'MEDIUM' if tests else 'HIGH', 'claims': positives})
    if negatives and production:
        findings.append({'id': 'documented-no-question-answering-claim-false', 'severity': 'HIGH', 'claims': negatives})
    return {'status': 'PASS' if not findings else 'FAIL', 'findings': findings, 'production_evidence': production, 'test_evidence': tests}
