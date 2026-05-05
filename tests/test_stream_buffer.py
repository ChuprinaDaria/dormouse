"""Тести для StreamBuffer — буферизація streaming для unsqueeze."""

from dormouse.stream_buffer import StreamBuffer


class TestStreamBuffer:
    def test_single_sentence(self):
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("Hello world."))
        results.extend(buf.flush())
        assert len(results) == 1
        assert results[0] == "Hello world."

    def test_two_sentences(self):
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("First sentence. Second sentence."))
        results.extend(buf.flush())
        assert len(results) == 2
        assert results[0] == "First sentence."
        assert results[1] == " Second sentence."

    def test_incremental_tokens(self):
        """Симулює streaming — по одному слову."""
        buf = StreamBuffer()
        results = []
        for token in ["Hello", " world", ".", " How", " are", " you", "?"]:
            results.extend(buf.feed(token))
        results.extend(buf.flush())
        assert len(results) == 2
        assert results[0] == "Hello world."
        assert results[1] == " How are you?"

    def test_newline_as_separator(self):
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("Line one\nLine two"))
        results.extend(buf.flush())
        assert len(results) == 2
        assert results[0] == "Line one\n"
        assert results[1] == "Line two"

    def test_exclamation_mark(self):
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("Wow! Great."))
        results.extend(buf.flush())
        assert len(results) == 2
        assert results[0] == "Wow!"

    def test_no_punctuation_flushes_at_end(self):
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("no punctuation here"))
        results.extend(buf.flush())
        assert len(results) == 1
        assert results[0] == "no punctuation here"

    def test_empty_feed(self):
        buf = StreamBuffer()
        results = list(buf.feed(""))
        assert results == []

    def test_none_feed(self):
        buf = StreamBuffer()
        results = list(buf.feed(None))
        assert results == []

    def test_flush_empty(self):
        buf = StreamBuffer()
        results = list(buf.flush())
        assert results == []

    def test_number_with_dot_not_split(self):
        """1.5 не має розрізати речення."""
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("Price is 1.5 dollars."))
        results.extend(buf.flush())
        assert len(results) == 1

    def test_ellipsis_as_separator(self):
        buf = StreamBuffer()
        results = []
        results.extend(buf.feed("Hmm... OK."))
        results.extend(buf.flush())
        assert len(results) == 2
