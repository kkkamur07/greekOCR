"""Calamari engine boundary.

The supported ``calamari-ocr`` distribution supplies its runtime; training
adapters live in :mod:`src.train` and the predictors that drive a trained
checkpoint live in :mod:`nomikos_inference.predictors`, so the project does not
vendor a second copy.
"""
