#!/usr/bin/python
#-*-coding:utf-8-*-
"""Run all tests
"""

import unittest
import test_pipeline
import test_dataset
import test_operators
import test_pytransformer
import test_reader_builder
import test_sharedqueue

if __name__ == '__main__':
    alltests = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(t) \
        for t in [
            test_pipeline.TestPipeline,
            test_dataset.TestDataset,
            test_operators.TestOperators,
            test_pytransformer.TestPyTransformer,
            test_reader_builder.TestReaderBuilder,
            test_sharedqueue.TestSharedQueue,
        ]
    ])

    was_succ = unittest\
                .TextTestRunner(verbosity=2)\
                .run(alltests)\
                .wasSuccessful()

    exit(0 if was_succ else 1)
