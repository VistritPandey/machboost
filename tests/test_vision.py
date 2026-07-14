from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from machboost.vision import (
    ContentAddressedVisionCache,
    VisualAssetStore,
    decode_data_url,
    normalize_multimodal_messages,
)


PNG_A = b"\x89PNG\r\n\x1a\nfirst-image"
PNG_B = b"\x89PNG\r\n\x1a\nsecond-image-with-a-different-size"


class VisionCacheTests(unittest.TestCase):
    def test_same_content_at_different_paths_reuses_features(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            first.write_bytes(PNG_A)
            second.write_bytes(PNG_A)
            cache = ContentAddressedVisionCache()

            cache.put(first, "projected-features")

            self.assertEqual(cache.get(second), "projected-features")
            self.assertEqual(cache.info().hits, 1)

    def test_modified_file_does_not_reuse_stale_features(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(PNG_A)
            cache = ContentAddressedVisionCache()
            cache.put(image, "old-features")

            image.write_bytes(PNG_B)

            self.assertIsNone(cache.get(image))
            self.assertEqual(cache.info().misses, 1)

    def test_lru_eviction_and_counters_are_stable(self):
        cache = ContentAddressedVisionCache(max_size=2)
        cache.put(b"one", 1)
        cache.put(b"two", 2)
        self.assertEqual(cache.get(b"one"), 1)
        cache.put(b"three", 3)

        self.assertIsNone(cache.get(b"two"))
        info = cache.info()
        self.assertEqual(info.size, 2)
        self.assertEqual(info.hits, 1)
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.puts, 3)
        self.assertEqual(info.evictions, 1)

    def test_composite_keys_preserve_image_order(self):
        cache = ContentAddressedVisionCache()
        self.assertNotEqual(cache.key_for([b"one", b"two"]), cache.key_for([b"two", b"one"]))


class VisualAssetStoreTests(unittest.TestCase):
    def test_data_url_and_raw_base64_share_content_addressed_path(self):
        encoded = base64.b64encode(PNG_A).decode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            store = VisualAssetStore(Path(directory))
            from_data_url = store.materialize(f"data:image/png;base64,{encoded}")
            from_base64 = store.materialize(encoded)

            self.assertEqual(from_data_url, from_base64)
            self.assertEqual(Path(from_data_url).read_bytes(), PNG_A)

    def test_remote_url_passes_through_without_download(self):
        store = VisualAssetStore()
        url = "https://example.com/image.png"
        self.assertEqual(store.materialize(url), url)

    def test_oversized_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VisualAssetStore(Path(directory), max_image_bytes=4)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                store.materialize(PNG_A)

    def test_data_url_parser_supports_percent_encoding(self):
        media_type, data = decode_data_url("data:image/png,%89PNG") or (None, None)
        self.assertEqual(media_type, "image/png")
        self.assertEqual(data, b"\x89PNG")


class MultimodalMessageTests(unittest.TestCase):
    def test_normalizes_ollama_images(self):
        messages, images = normalize_multimodal_messages(
            [{"role": "user", "content": "What is shown?", "images": ["first", "second"]}]
        )

        self.assertEqual(messages, [{"role": "user", "content": "What is shown?"}])
        self.assertEqual(images, ["first", "second"])

    def test_normalizes_openai_image_parts(self):
        messages, images = normalize_multimodal_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        )

        self.assertEqual(messages[0]["content"], "Read this.")
        self.assertEqual(images, ["data:image/png;base64,AAAA"])

    def test_rejects_unknown_multimodal_part(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            normalize_multimodal_messages(
                [{"role": "user", "content": [{"type": "audio_url", "audio_url": "sound.wav"}]}]
            )


if __name__ == "__main__":
    unittest.main()
