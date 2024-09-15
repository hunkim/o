from tokenizers import Tokenizer
MAX_TOKENS = 2048

solar_tokenizer = Tokenizer.from_pretrained("upstage/solar-pro-preview-tokenizer")


def num_of_tokens(text):
    return len(solar_tokenizer.encode(text).ids)


def limit_chat_history(chat_history, max_tokens=MAX_TOKENS):
    limited_history = []
    total_length = 0
    for message in reversed(chat_history):
        message_length = num_of_tokens(message.content)
        if total_length + message_length > max_tokens:
            break
        limited_history.insert(0, message)
        total_length += message_length
    return limited_history
