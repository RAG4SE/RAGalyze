import unittest
from ragalyze.rag.splitter.utf8 import find_safe_utf8_boundary


class TestUTF8Boundary(unittest.TestCase):

    def test_ascii_text(self):
        # Test with simple ASCII text
        data = b"Hello world, this is a test"
        pos = 15  # Should be in the middle of "world"
        safe_pos = find_safe_utf8_boundary(data, pos)
        # Should return the same position for ASCII
        self.assertEqual(safe_pos, pos)

    def test_utf8_text(self):
        # Test with UTF-8 text containing multi-byte characters
        data = "Hello 世界, this is a test".encode("utf-8")
        pos = 10  # Position that might be in the middle of a UTF-8 character
        safe_pos = find_safe_utf8_boundary(data, pos)
        # Should return a position that's at the start of a UTF-8 character
        self.assertTrue(safe_pos <= pos)
        # Verify we can decode the text up to the safe position
        try:
            decoded = data[:safe_pos].decode("utf-8")
        except UnicodeDecodeError:
            self.fail("Safe position is not actually safe for UTF-8 decoding")

    def test_edge_cases(self):
        # Test edge cases
        data = "Hello 世界".encode("utf-8")

        # Position at the very end
        safe_pos = find_safe_utf8_boundary(data, len(data))
        self.assertEqual(safe_pos, len(data))

        # Position at the very beginning
        safe_pos = find_safe_utf8_boundary(data, 0)
        self.assertEqual(safe_pos, 0)

        # Position beyond the data
        safe_pos = find_safe_utf8_boundary(data, len(data) + 10)
        self.assertEqual(safe_pos, len(data))


if __name__ == "__main__":
    unittest.main()
