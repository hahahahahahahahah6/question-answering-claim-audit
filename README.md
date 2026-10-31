# question-answering-claim-audit

An offline, standard-library-only static auditor for tools that promise to find answers in documents. It compares prose with provider-resolved AST evidence; it never executes the inspected source or imports its dependencies.

Install and run with Python 3.9 or later:

```sh
python3 -m pip install .
question-answering-claim-audit /path/to/project
python3 -m question_answering_claim_audit /path/to/project --json
python3 -m unittest discover -s tests -v
```

The root README.md and Markdown files beneath any docs path component are inspected. Fenced blocks are excluded; inline code counts. Paragraphs are normalised independently. Negative spans are removed before positive phrases are matched, longest first. Source evidence comes from Transformers task calls or pipeline construction, and bound Haystack reader calls. Retrieval, string matching, regular expressions, generic chat, and model construction alone do not establish an answer-finding operation.

Symlinked files and directories are excluded. Source beneath fixtures, docs, and examples components is excluded. Files beneath tests, or named test_*.py or *_test.py, provide test evidence separately. Invalid syntax provides no evidence. Resolution is static within each module; dynamic imports and runtime behaviour cannot be inferred.

```text
HIGH documented-question-answering-claim-false
MEDIUM documented-question-answering-claim-false (test evidence only)
HIGH documented-no-question-answering-claim-false
```

Exit status is 0 for a clean result, 2 for a high severity finding, and 1 for input or usage errors. Test evidence alone produces a medium finding and status 0 unless --fail-on MEDIUM is selected. A positive promise with no source produces a high finding; without a promise an empty project passes.

Options: --json emits structured results, --verbose includes evidence paths and matched phrases, --ignore GLOB excludes matching paths or components (repeatable), --fail-on HIGH|MEDIUM selects the failure threshold, and --version prints 0.1.0. JSON includes status, findings, production_evidence, and test_evidence; each finding includes id, severity, and claims with path and phrase.
