"""Independent benchmark metric and dataset validation tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks.evaluate_ocr import (
    BenchmarkCase,
    edit_distance,
    evaluate_cases,
    environment_info,
    load_cases,
    normalize_alnum,
    normalize_strict,
    score_text,
    summarize,
)


class MetricTests(unittest.TestCase):
    def test_levenshtein_known_distances(self):
        for reference, prediction, expected in (
            ("", "", 0), ("abc", "", 3), ("", "abc", 3),
            ("kitten", "sitting", 3), ("serial123", "serial123", 0),
            ("12345", "123X5", 1), ("12", "21", 2),
        ):
            with self.subTest(reference=reference, prediction=prediction):
                self.assertEqual(edit_distance(reference, prediction), expected)
                self.assertEqual(edit_distance(prediction, reference), expected)

    def test_normalization_retains_strict_punctuation(self):
        self.assertEqual(normalize_strict('  ＡＢ-１２\n\tRev A1 HT#  '), 'ab-12 rev a1 ht#')
        self.assertEqual(normalize_alnum('  ＡＢ-１２\n\tRev A1 HT#  '), 'ab12reva1ht')
        self.assertEqual(normalize_strict('Straße'), 'strasse')
        scores = score_text('AB-12', 'ab 12')
        self.assertEqual(scores['strict']['character_edits'], 1)
        self.assertEqual(scores['alnum']['character_edits'], 0)
        self.assertFalse(scores['strict']['exact_match'])
        self.assertTrue(scores['alnum']['exact_match'])

    def test_cer_counts_insertions_and_can_exceed_one(self):
        self.assertEqual(score_text('1', '1234')['alnum']['cer'], 3)
        self.assertEqual(score_text('', '')['alnum']['cer'], 0)
        self.assertEqual(score_text('', 'AB')['alnum']['cer'], 2)

    def test_aggregate_uses_corpus_character_denominator(self):
        results = [
            {'metrics': score_text('AB', 'AB'), 'seconds': 1.25, 'error': None},
            {'metrics': score_text('12345678', ''), 'seconds': 2.75, 'error': 'Unreadable'},
        ]
        summary = summarize(results)
        self.assertEqual(summary['image_count'], 2)
        self.assertEqual(summary['failed_images'], 1)
        self.assertEqual(summary['total_image_seconds'], 4)
        for metric in summary['metrics'].values():
            self.assertEqual(metric['corpus_cer'], 0.8)
            self.assertEqual(metric['mean_image_cer'], 0.5)
            self.assertEqual(metric['exact_match_rate'], 0.5)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = self.root / 'labels.txt'

    def tearDown(self):
        self.temp.cleanup()

    def test_label_split_preserves_semicolon_in_transcription(self):
        self.labels.write_text('\ufeffid_img1; AB;12 REV A\n\n', encoding='utf-8')
        image = self.root / 'ID_IMG1.JPG'
        image.touch()
        self.assertEqual(load_cases(self.labels, self.root), [BenchmarkCase('id_img1', 'AB;12 REV A', image)])

    def test_missing_image_is_explicit_error(self):
        self.labels.write_text('id_img1; AB12\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'No image found'):
            load_cases(self.labels, self.root)

    def test_duplicate_casefolded_label_is_rejected(self):
        self.labels.write_text('ID_IMG1; AB12\nid_img1; CD34\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate image ID'):
            load_cases(self.labels, self.root)

    def test_duplicate_image_stems_are_rejected(self):
        self.labels.write_text('id_img1; AB12\n', encoding='utf-8')
        (self.root / 'id_img1.png').touch()
        (self.root / 'id_img1.jpg').touch()
        with self.assertRaisesRegex(ValueError, 'Multiple images'):
            load_cases(self.labels, self.root)

    def test_invalid_or_empty_labels_are_rejected(self):
        for contents in ('', 'id_img1 AB12', 'id_img1;', ';AB12', '../id_img1; AB12'):
            with self.subTest(contents=contents):
                self.labels.write_text(contents, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_cases(self.labels, self.root)


class EvaluationTests(unittest.TestCase):
    def test_environment_records_both_backends_and_source_hashes(self):
        environment = environment_info()
        self.assertIn('rapidocr', environment['packages'])
        self.assertIn('onnxruntime', environment['packages'])
        self.assertEqual(environment['ocr_engine_sha256'], environment['source_sha256']['ocr_engine.py'])
        self.assertEqual(len(environment['source_sha256']['ocr_readers.py']), 64)

    def test_runs_without_passing_labels_and_scores_failures(self):
        cases = [
            BenchmarkCase('first', 'AB-12', Path('first.jpg')),
            BenchmarkCase('second', '1234', Path('second.jpg')),
        ]
        engine = Mock()
        engine.analyze.side_effect = [{'text': 'ab 12'}, RuntimeError('No model')]
        loader = Mock(side_effect=['first image pixels', 'second image pixels'])
        options = SimpleNamespace(mode='all')
        ticks = iter([10, 11, 11, 13])
        results = evaluate_cases(
            cases, engine, options, image_loader=loader,
            text_extractor=lambda payload: payload['text'], clock=lambda: next(ticks), progress=lambda _: None,
        )
        self.assertEqual(engine.analyze.call_args_list[0].args, ('first image pixels', options))
        self.assertEqual(engine.analyze.call_args_list[1].args, ('second image pixels', options))
        self.assertEqual(results[0]['prediction'], 'ab 12')
        self.assertTrue(results[0]['metrics']['alnum']['exact_match'])
        self.assertEqual(results[0]['seconds'], 1)
        self.assertEqual(results[1]['error'], 'RuntimeError: No model')
        self.assertEqual(results[1]['prediction'], '')
        self.assertEqual(results[1]['metrics']['alnum']['cer'], 1)
        self.assertEqual(summarize(results)['metrics']['alnum']['corpus_cer'], 0.5)


if __name__ == '__main__':
    unittest.main()
