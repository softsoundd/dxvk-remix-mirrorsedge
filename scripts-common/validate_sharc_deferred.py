#!/usr/bin/env python3
"""Compare deferred SHARC segments with independent forward per-vertex propagation.

Models radiance before/after insertion, multiple attenuation events between cached
vertices, failed insertions, repeated cache cells, and zero color channels. The
shader integration validator separately checks the actual compiled variants.
"""
import math
import random
import unittest


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def mul(a, b):
    return tuple(x * y for x, y in zip(a, b))


ZERO = (0., 0., 0.)
ONE = (1., 1., 1.)


def forward(events):
    vertices = []
    for kind, value in events:
        if kind == 'insert':
            if value is not None:
                vertices.append([value, ZERO, ONE])
        elif kind == 'light':
            for v in vertices:
                v[1] = add(v[1], mul(v[2], value))
        else:
            for v in vertices:
                v[2] = mul(v[2], value)
    return [(v[0], v[1]) for v in vertices]


def deferred(events, depth):
    segments = []
    for kind, value in events:
        if kind == 'insert':
            if value is not None and len(segments) < depth:
                segments.append([value, ZERO, ONE])
        elif segments:
            v = segments[-1]
            if kind == 'light':
                v[1] = add(v[1], mul(v[2], value))
            else:
                v[2] = mul(v[2], value)
    result = []
    suffix = ZERO
    for key, local, weight in reversed(segments):
        suffix = add(local, mul(weight, suffix))
        result.append((key, suffix))
    return list(reversed(result))


def deferred_fixed(events, depth):
    keys, local, weights = [None] * depth, [ZERO] * depth, [ONE] * depth
    count = 0
    for kind, value in events:
        if kind == 'insert':
            if value is not None and count < depth:
                for i in range(depth):
                    if count == i:
                        keys[i], local[i], weights[i] = value, ZERO, ONE
                count += 1
        else:
            for i in range(depth):
                if count == i + 1:
                    if kind == 'light':
                        local[i] = add(local[i], mul(weights[i], value))
                    else:
                        weights[i] = mul(weights[i], value)
    result, suffix = [], ZERO
    for i in reversed(range(depth)):
        if i < count:
            suffix = add(local[i], mul(weights[i], suffix))
            result.append((keys[i], suffix))
    return list(reversed(result))


class DeferredSharcTests(unittest.TestCase):
    def compare(self, events, depth=8):
        expected, actual = forward(events), deferred(events, depth)
        self.assertEqual(actual, deferred_fixed(events, depth))
        self.assertEqual(len(expected), len(actual))
        for (key1, a), (key2, b) in zip(expected, actual):
            self.assertEqual(key1, key2)
            for x, y in zip(a, b):
                self.assertTrue(math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10), (x, y))

    def test_emission_before_insertion_and_zero_channels(self):
        events = [('light', (999., 999., 999.)), ('weight', (.1, .2, .3)),
                  ('insert', 0), ('light', (1., 2., 3.)),
                  ('weight', (0., .5, 2.)), ('light', (4., 5., 6.)),
                  ('insert', 1), ('light', (2., 1., 4.))]
        self.compare(events)
        self.assertEqual(deferred(events, 8), [(0, (1., 5., 23.)), (1, (2., 1., 4.))])

    def test_failed_insertion_keeps_prior_estimate(self):
        self.compare([('insert', 7), ('weight', (.2, .5, .8)),
                      ('insert', None), ('light', (3., 2., 1.)),
                      ('weight', (.5, 0., .4)), ('insert', 9), ('light', (5., 6., 7.))])

    def test_repeated_cells_preserve_sample_counts(self):
        events = [('insert', 2), ('light', ONE), ('weight', (.5, .5, .5)),
                  ('insert', 2), ('light', ONE)]
        self.compare(events)
        results = deferred(events, 8)
        self.assertEqual(sum(key == 2 for key, _ in results), 2)
        self.assertEqual(add(results[0][1], results[1][1]), (2.5, 2.5, 2.5))

    def test_empty_and_zero_light_paths(self):
        self.compare([])
        self.compare([('weight', ZERO), ('light', ONE)])
        self.compare([('insert', 0), ('weight', ZERO), ('insert', 1)])

    def test_fixed_depth_ignores_overflow_and_unwritten_slots(self):
        for depth in (4, 8):
            for count in range(depth + 3):
                events = [(kind, value) for key in range(count)
                          for kind, value in [('insert', key), ('light', ONE), ('weight', (.5, 0., 2.))]]
                self.assertEqual(deferred(events, depth), deferred_fixed(events, depth))

    def test_random_paths_at_both_compiled_depths(self):
        rng = random.Random(9122026)
        for depth in (4, 8):
            for _ in range(1000):
                events = [('light', (13., 17., 19.)), ('weight', (.1, .9, .5))]
                for bounce in range(depth):
                    events += [('light', tuple(rng.random() * 10 for _ in range(3))),
                               ('insert', rng.randrange(4) if rng.random() < .75 else None)]
                    for _ in range(rng.randrange(1, 5)):
                        events.append(('light', tuple(rng.random() * 20 for _ in range(3))))
                        events.append(('weight', tuple(rng.choice((0., .1, .5, 1., 2.)) for _ in range(3))))
                events.append(('light', tuple(rng.random() * 10 for _ in range(3))))
                self.compare(events, depth)


if __name__ == '__main__':
    unittest.main()
