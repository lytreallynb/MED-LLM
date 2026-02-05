"""Tests for the LLM module."""
import pytest
from unittest.mock import patch, MagicMock
from llm import QwenConfig, QwenLLM


class TestQwenConfig:
    def test_default_values(self):
        config = QwenConfig()
        assert config.model_name == "qwen2.5-72b-instruct"
        assert config.timeout == 60
        assert config.temperature == 0.7

    def test_custom_values(self):
        config = QwenConfig(
            model_name="qwen2.5-7b-instruct",
            temperature=0.5,
            max_tokens=512,
        )
        assert config.model_name == "qwen2.5-7b-instruct"
        assert config.temperature == 0.5
        assert config.max_tokens == 512


class TestQwenLLM:
    def test_requires_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                QwenLLM(QwenConfig(api_key=None))
            assert "DASHSCOPE_API_KEY" in str(exc_info.value)

    def test_accepts_api_key_in_config(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)
        assert llm.api_key == "test_key"

    def test_build_messages_simple(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        messages = llm._build_messages("What is aspirin?")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is aspirin?"

    def test_build_messages_with_history(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = llm._build_messages("New question", history=history)

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["content"] == "Previous answer"
        assert messages[3]["content"] == "New question"

    def test_build_messages_custom_system(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        custom_system = "You are a helpful assistant."
        messages = llm._build_messages("Hello", system_prompt=custom_system)

        assert messages[0]["content"] == custom_system

    def test_parse_response_output_text(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        data = {"output": {"text": "Test response"}}
        result = llm._parse_response(data)
        assert result == "Test response"

    def test_parse_response_output_choices(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        data = {
            "output": {
                "choices": [
                    {"message": {"content": "Test response from choices"}}
                ]
            }
        }
        result = llm._parse_response(data)
        assert result == "Test response from choices"

    def test_parse_response_choices_format(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        data = {
            "choices": [
                {"message": {"content": "Direct choices response"}}
            ]
        }
        result = llm._parse_response(data)
        assert result == "Direct choices response"

    def test_parse_response_invalid_format(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        data = {"unexpected": "format"}
        with pytest.raises(ValueError) as exc_info:
            llm._parse_response(data)
        assert "Unexpected response format" in str(exc_info.value)

    @patch('llm.requests.post')
    def test_generate_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": {"text": "Generated text"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        result = llm.generate("Test prompt")
        assert result == "Generated text"

    def test_chat_with_context(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        with patch.object(llm, 'generate', return_value="Mocked response") as mock_gen:
            result = llm.chat("What is aspirin?", context="Aspirin is a pain reliever.")

            assert result == "Mocked response"
            call_args = mock_gen.call_args[0][0]
            assert "FDA drug information" in call_args
            assert "Aspirin is a pain reliever" in call_args
            assert "What is aspirin?" in call_args

    def test_chat_without_context(self):
        config = QwenConfig(api_key="test_key")
        llm = QwenLLM(config)

        with patch.object(llm, 'generate', return_value="Mocked response") as mock_gen:
            result = llm.chat("Simple question")

            assert result == "Mocked response"
            mock_gen.assert_called_once_with("Simple question")
