import os
import unittest

from Experiments.semantic_search_pgVector import SemanticSearchExperiment


class MyTestCase(unittest.TestCase):

    experiment = None
    test_list = [
        "Windows Hyper-V NT Kernel Integration VSP received three patches",
        "block IP addresses from Russia and China on my Ubiquiti",
        "brute-forcing TOTP",
        "One question to ask is whether there might be any shorter route for brute forcing a solution",
        "1.2 X 1,000,000^4",
        "HMAC algorithm" # 2 instances
    ]

    @classmethod
    def setUpClass(cls):
        cls.experiment = SemanticSearchExperiment()
        cls.experiment.initialize_data()

    def test_for_expected_results_from_semantic_search(self, ):
        test_phrase = "Windows Hyper-V NT Kernel Integration VSP received three patches"

        splits = self.experiment.semantic_search(test_phrase)

        found_it = False

        split_count=0

        for split in splits:
            print(split)
            if test_phrase in split.page_content:
                found_it = True
                print(f'\nFound expected results at split {split_count} \n')
                break
        split_count = split_count + 1

        self.assertTrue(found_it, f'Expected results not found. Test phrase: {test_phrase}')


if __name__ == '__main__':
    unittest.main()
