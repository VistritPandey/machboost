import unittest

from machboost.ollama_compat import (
    apply_generate_template,
    normalize_ollama_options,
    structured_output_instruction,
    truncate_messages,
    truncate_prompt,
    validate_structured_output,
)


class WordService:
    def encode(self, text, **_kwargs):
        return tuple(text.split())

    def decode(self, tokens, **_kwargs):
        return " ".join(tokens)


def render(messages):
    return " ".join(str(message.get("content") or "") for message in messages)


class OllamaOptionTests(unittest.TestCase):
    def test_top_level_and_nested_options_are_normalized(self):
        result = normalize_ollama_options(
            {
                "num_ctx": "4096",
                "temperature": 0.5,
                "stop": "END",
                "think": "high",
                "format": "json",
                "options": {"top_p": 0.9},
            }
        )
        self.assertEqual(result["num_ctx"], 4096)
        self.assertEqual(result["stop"], ["END"])
        self.assertEqual(result["_think"], "high")
        self.assertEqual(result["_format"], "json")

    def test_invalid_sampling_and_format_are_rejected(self):
        for payload in (
            {"num_ctx": 0},
            {"top_p": 1.5},
            {"temperature": -1},
            {"stop": ["ok", 3]},
            {"format": "yaml"},
            {"think": 3},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                normalize_ollama_options(payload)

    def test_null_options_are_treated_as_omitted(self):
        result = normalize_ollama_options(
            {
                "num_ctx": None,
                "options": {"num_predict": 128, "top_k": None},
            }
        )

        self.assertEqual(result, {"num_predict": 128})


class ContextWindowTests(unittest.TestCase):
    def setUp(self):
        self.service = WordService()

    def test_prompt_left_truncation_respects_num_keep(self):
        prompt, removed = truncate_prompt(
            self.service,
            "one two three four five six seven eight",
            num_ctx=7,
            max_tokens=2,
            num_keep=2,
        )
        self.assertEqual(prompt, "one two six seven eight")
        self.assertEqual(removed, 3)

    def test_prompt_can_fail_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "requires 8 tokens"):
            truncate_prompt(
                self.service,
                "one two three four five six seven eight",
                num_ctx=7,
                max_tokens=2,
                truncate=False,
            )

    def test_chat_drops_oldest_turns_and_keeps_system_and_latest(self):
        messages, removed = truncate_messages(
            self.service,
            [
                {"role": "system", "content": "system rule"},
                {"role": "user", "content": "old question words"},
                {"role": "assistant", "content": "old answer words"},
                {"role": "user", "content": "latest question"},
            ],
            render=render,
            num_ctx=7,
            max_tokens=2,
        )
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "latest question")
        self.assertGreater(removed, 0)

    def test_system_content_that_exceeds_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "system messages"):
            truncate_messages(
                self.service,
                [{"role": "system", "content": "one two three four five"}],
                render=render,
                num_ctx=3,
                max_tokens=1,
            )


class TemplateAndFormatTests(unittest.TestCase):
    def test_generate_template_replaces_ollama_placeholders(self):
        result = apply_generate_template(
            "hello",
            {"_system": "be brief", "_template": "S={{ .System }} P={{ .Prompt }} A={{ .Response }}"},
        )
        self.assertEqual(result, "S=be brief P=hello A=")

    def test_json_schema_instruction_and_validation(self):
        schema = {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }
        instruction = structured_output_instruction(schema)
        value = validate_structured_output('{"answer":"yes"}', schema)

        self.assertIn("JSON Schema", instruction)
        self.assertEqual(value, {"answer": "yes"})
        with self.assertRaisesRegex(ValueError, "missing answer"):
            validate_structured_output("{}", schema)
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            validate_structured_output("not-json", "json")


if __name__ == "__main__":
    unittest.main()
