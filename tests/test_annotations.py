import unittest

from microphaselab.annotations import AnnotationFormatError, parse_polygon


class ParsePolygonTests(unittest.TestCase):
    def test_python_literal(self):
        self.assertEqual(
            parse_polygon("[(1, 2), (5, 2), (5, 8)]"),
            [(1.0, 2.0), (5.0, 2.0), (5.0, 8.0)],
        )

    def test_json_geojson_and_wkt(self):
        expected = [(1.0, 2.0), (5.0, 2.0), (5.0, 8.0)]
        self.assertEqual(parse_polygon("[[1, 2], [5, 2], [5, 8]]"), expected)
        self.assertEqual(
            parse_polygon({"type": "Polygon", "coordinates": [[[1, 2], [5, 2], [5, 8], [1, 2]]]}),
            expected,
        )
        self.assertEqual(parse_polygon("POLYGON ((1 2, 5 2, 5 8, 1 2))"), expected)

    def test_rejects_code_and_bad_geometry(self):
        with self.assertRaises(AnnotationFormatError):
            parse_polygon("__import__('os').system('echo unsafe')")
        with self.assertRaises(AnnotationFormatError):
            parse_polygon("[(1, 2), (1, 2), (1, 2)]")
        with self.assertRaises(AnnotationFormatError):
            parse_polygon("[(1, 2), (3, float('nan')), (5, 6)]")


if __name__ == "__main__":
    unittest.main()
