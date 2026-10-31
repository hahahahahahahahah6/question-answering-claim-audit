import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from question_answering_claim_audit.audit import audit, claims, evidence
from question_answering_claim_audit.phrases import POSITIVE, NEGATIVE
from question_answering_claim_audit.__main__ import main


class EvidenceTests(unittest.TestCase):
    pass


GOOD = {
    'direct': 'import transformers\ntransformers.pipeline("question-answering")',
    'alias_module': 'import transformers as tf\ntf.pipeline("question_answering")',
    'from_import': 'from transformers import pipeline\npipeline("question-answering")',
    'alias_function': 'from transformers import pipeline as p\np("question-answering")',
    'keyword': 'from transformers import pipeline\npipeline(task="question-answering")',
    'keyword_alias': 'import transformers as t\nt.pipeline(model="m", task="QUESTION_ANSWERING")',
    'suffix': 'from transformers import pipeline\npipeline("question-answering_en")',
    'boundary_digit': 'from transformers import pipeline\npipeline("question-answering2")',
    'qa_direct': 'import transformers\ntransformers.QuestionAnsweringPipeline()',
    'qa_alias_module': 'import transformers as t\nt.QuestionAnsweringPipeline()',
    'qa_from': 'from transformers import QuestionAnsweringPipeline\nQuestionAnsweringPipeline()',
    'qa_alias': 'from transformers import QuestionAnsweringPipeline as Q\nQ()',
    'reader_plain_call': 'from haystack.nodes import FARMReader\nr=FARMReader()\nr()',
    'reader_bound': 'from haystack.nodes import FARMReader\nr=FARMReader()\nr.predict(question="q")',
    'reader_chained': 'from haystack.nodes import FARMReader\nFARMReader().predict()',
    'reader_module': 'import haystack as h\nr=h.TransformersReader()\nr.run()',
    'reader_alias': 'from haystack.components.readers import ExtractiveReader as R\nr=R()\nr.run()',
    'reader_self': 'from haystack.nodes import FARMReader\nclass C:\n def __init__(self):\n  self.reader=FARMReader()\n  self.reader.predict()',
    'reader_cross_method': 'from haystack.nodes import FARMReader\nclass C:\n def __init__(self):\n  self.reader=FARMReader()\n def run(self):\n  self.reader.predict()',
    'reader_annotated': 'from haystack.nodes import FARMReader\nr: object=FARMReader()\nr.predict()',
    'local_import': 'def run():\n from transformers import pipeline as p\n return p("question-answering")',
    'local_reader_import': 'def run():\n from haystack.nodes import FARMReader\n r=FARMReader()\n return r.predict()',
    'inside_function': 'from transformers import pipeline\ndef run():\n return pipeline("question-answering")',
}
BAD = {
    'retrieval': 'results = db.search("question-answering")',
    'bm25': 'BM25Retriever().retrieve("q")',
    'faq': 'faq={"q":"a"}\nanswer=faq.get("q")',
    'regex': 'import re\nre.search("answer: (.*)", text)',
    'chat': 'client.chat.completions.create(model="m", messages=[])',
    'generate': 'model.generate("q")',
    'question_generation': 'from transformers import pipeline\npipeline("question-generation")',
    'bare_import': 'from transformers import pipeline, QuestionAnsweringPipeline',
    'bare_reader_import': 'from haystack.nodes import FARMReader',
    'shadow_function': 'from transformers import pipeline\ndef pipeline(x): pass\npipeline("question-answering")',
    'shadow_class': 'from transformers import QuestionAnsweringPipeline\nclass QuestionAnsweringPipeline: pass\nQuestionAnsweringPipeline()',
    'shadow_module': 'import transformers\ntransformers=object()\ntransformers.pipeline("question-answering")',
    'shadow_parameter': 'from transformers import pipeline\ndef f(pipeline):\n pipeline("question-answering")',
    'shadow_reader': 'from haystack.nodes import FARMReader\nclass FARMReader: pass\nr=FARMReader()\nr.predict()',
    'string': 's="transformers.pipeline(\'question-answering\')"',
    'subprocess': 'import subprocess\nsubprocess.run(["answerer", "question-answering"])',
    'construction': 'from haystack.nodes import FARMReader\nr=FARMReader()',
    'reader_reassigned': 'from haystack.nodes import FARMReader\nr=FARMReader()\nr=object()\nr.predict()',
    'reader_parameter_shadow': 'from haystack.nodes import FARMReader\nr=FARMReader()\ndef f(r):\n r.predict()',
    'local_import_does_not_leak': 'def setup():\n from transformers import pipeline\npipeline("question-answering")',
    'unbound': 'reader.predict("q")',
    'unbound_pipeline': 'pipeline("question-answering")',
    'unbound_class_call': 'from haystack.nodes import FARMReader\nFARMReader.predict("q")',
    'other_provider': 'from fake import pipeline, QuestionAnsweringPipeline\npipeline("question-answering")\nQuestionAnsweringPipeline()',
    'other_reader_provider': 'from fake import FARMReader\nr=FARMReader()\nr.predict()',
    'automodel': 'from transformers import AutoModelForQuestionAnswering\nAutoModelForQuestionAnswering.from_pretrained("m")',
    'prefix': 'from transformers import pipeline\npipeline("requestion answering")',
    'plural': 'from transformers import pipeline\npipeline("question answerings")',
    'dynamic_task': 'from transformers import pipeline\ntask="question-answering"\npipeline(task)',
    'syntax': 'from transformers import pipeline\n(',
    'comment': '# transformers.pipeline("question-answering")',
}
for task in ['summarization', 'translation', 'text-classification', 'sentiment-analysis', 'text-generation', 'fill-mask', 'token-classification', 'feature-extraction']:
    BAD[task] = 'from transformers import pipeline\npipeline('+repr(task)+')'
for reader in ['FARMReader', 'TransformersReader', 'ExtractiveReader']:
    GOOD['reader_'+reader] = 'from haystack.nodes import '+reader+'\nr='+reader+'()\nr.predict()'
    BAD['uncalled_'+reader] = 'from haystack.nodes import '+reader+'\nr='+reader+'()'


def evidence_test(source, expected):
    def test(self):
        self.assertEqual(evidence(source), expected)
    return test


for name, source in GOOD.items():
    setattr(EvidenceTests, 'test_good_'+name, evidence_test(source, True))
for name, source in BAD.items():
    setattr(EvidenceTests, 'test_bad_'+name, evidence_test(source, False))


class PhraseTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(POSITIVE), 50)
        self.assertEqual(len(NEGATIVE), 26)
        self.assertNotIn('question answering', POSITIVE)

    def test_longest(self):
        self.assertEqual(claims('answers questions from documents'), (['answers questions from documents'], []))

    def test_negative_overlap(self):
        self.assertEqual(claims('does not support question answering'), ([], ['does not support question answering']))

    def test_inline(self):
        self.assertEqual(claims('`answers questions`')[0], ['answers questions'])

    def test_fences(self):
        for marker in ['```', '~~~', '   ```', '  ~~~~']:
            with self.subTest(marker=marker):
                self.assertEqual(claims(marker+'text\nanswers questions\n'+marker.strip()+'\n'), ([], []))

    def test_fence_length(self):
        self.assertEqual(claims('````\n```\nanswers questions\n````'), ([], []))

    def test_unclosed_fence(self):
        self.assertEqual(claims('```\nanswers questions'), ([], []))

    def test_paragraphs(self):
        self.assertEqual(claims('answers\n\nquestions'), ([], []))

    def test_normalization(self):
        self.assertEqual(claims("DOESN'T answer user-questions"), ([], ['does not answer user questions']))

    def test_boundary(self):
        self.assertEqual(claims('reanswers questions answers questionsx'), ([], []))


def phrase_test(phrase, negative):
    def test(self):
        pos, neg = claims(phrase)
        self.assertEqual((pos, neg), ([], [phrase]) if negative else ([phrase], []))
    return test


for index, phrase in enumerate(POSITIVE):
    setattr(PhraseTests, 'test_positive_%02d' % index, phrase_test(phrase, False))
for index, phrase in enumerate(NEGATIVE):
    setattr(PhraseTests, 'test_negative_%02d' % index, phrase_test(phrase, True))


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(args))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_pass(self):
        self.write('README.md', 'answers questions')
        self.write('app.py', GOOD['direct'])
        self.assertEqual(self.cli(str(self.root))[0], 0)
        self.assertEqual(audit(self.root)['status'], 'PASS')

    def test_high(self):
        self.write('README.md', 'answers questions')
        self.write('app.py', BAD['faq'])
        result = audit(self.root)
        self.assertEqual(result['findings'][0]['id'], 'documented-question-answering-claim-false')
        self.assertEqual(result['findings'][0]['severity'], 'HIGH')
        self.assertEqual(self.cli(str(self.root))[0], 2)

    def test_no_source_asymmetry(self):
        self.write('README.md', 'answers questions')
        self.assertEqual(self.cli(str(self.root))[0], 2)
        self.write('README.md', 'no question answering')
        self.assertEqual(self.cli(str(self.root))[0], 0)

    def test_no_claim(self):
        self.assertEqual(audit(self.root)['status'], 'PASS')
        self.write('app.py', GOOD['direct'])
        self.assertEqual(audit(self.root)['status'], 'PASS')

    def test_negative_high(self):
        self.write('README.md', 'no question answering')
        self.write('app.py', GOOD['direct'])
        result = audit(self.root)
        self.assertEqual(len(result['findings']), 1)
        self.assertEqual(result['findings'][0]['id'], 'documented-no-question-answering-claim-false')
        self.assertEqual(result['findings'][0]['severity'], 'HIGH')
        self.assertEqual(self.cli(str(self.root))[0], 2)

    def test_one_per_family(self):
        self.write('README.md', 'answers questions\n\nanswers faqs')
        self.write('docs/a.md', 'answers user questions')
        self.assertEqual(len(audit(self.root)['findings']), 1)

    def test_test_only(self):
        for name in ['tests/unit.py', 'test_app.py', 'app_test.py']:
            with self.subTest(name=name):
                path = self.write(name, GOOD['direct'])
                self.write('README.md', 'answers questions')
                self.assertEqual(audit(self.root)['findings'][0]['severity'], 'MEDIUM')
                self.assertEqual(self.cli(str(self.root))[0], 0)
                self.assertEqual(self.cli(str(self.root), '--fail-on', 'MEDIUM')[0], 2)
                path.unlink()

    def test_negative_test_only(self):
        self.write('README.md', 'no question answering')
        self.write('tests/test_app.py', GOOD['direct'])
        self.assertEqual(audit(self.root)['status'], 'PASS')

    def test_scope(self):
        self.write('nested/README.md', 'answers questions')
        self.write('guide.md', 'answers questions')
        self.write('docs/guide.txt', 'answers questions')
        self.assertEqual(audit(self.root)['status'], 'PASS')
        self.write('nested/docs/sub/guide.md', 'answers questions')
        self.assertEqual(self.cli(str(self.root))[0], 2)

    def test_exclusions(self):
        self.write('README.md', 'answers questions')
        for folder in ['docs', 'examples', 'fixtures']:
            self.write(folder+'/nested/app.py', GOOD['direct'])
            self.write(folder+'/tests/test_app.py', GOOD['direct'])
        self.assertEqual(audit(self.root)['findings'][0]['severity'], 'HIGH')

    def test_module_isolation(self):
        self.write('README.md', 'answers questions')
        self.write('a.py', 'from transformers import pipeline')
        self.write('b.py', 'pipeline("question-answering")')
        self.assertEqual(self.cli(str(self.root))[0], 2)

    def test_symlinks(self):
        self.write('README.md', 'answers questions')
        target = self.write('fixtures/app.py', GOOD['direct'])
        (self.root/'link.py').symlink_to(target)
        (self.root/'linked').symlink_to(target.parent, target_is_directory=True)
        self.assertEqual(self.cli(str(self.root))[0], 2)

    def test_ignore(self):
        self.write('README.md', 'answers questions')
        self.write('app.py', GOOD['direct'])
        self.assertEqual(self.cli(str(self.root), '--ignore', '*.py')[0], 2)
        self.assertEqual(self.cli(str(self.root), '--ignore', 'README.md', '--ignore', '*.py')[0], 0)

    def test_json_verbose(self):
        self.write('README.md', 'answers questions')
        code, out, _ = self.cli(str(self.root), '--json')
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out), audit(self.root))
        self.assertIn('production_evidence', self.cli(str(self.root), '--verbose')[1])

    def test_version(self):
        self.assertEqual(self.cli('--version')[:2], (0, '0.1.0\n'))

    def test_usage(self):
        for args in [(), ('/no/such/path',), (str(self.root), '--unknown'), (str(self.root), '--fail-on', 'LOW')]:
            with self.subTest(args=args):
                self.assertEqual(self.cli(*args)[0], 1)

    def test_readme_has_no_claims(self):
        readme = Path(__file__).resolve().parents[1]/'README.md'
        self.assertEqual(claims(readme.read_text()), ([], []))


if __name__ == '__main__':
    unittest.main()
