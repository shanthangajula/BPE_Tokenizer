from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

        # Initialize byte vocabulary
        for i in range(256):
            self.vocab[i] = bytes([i])

    def train(self, corpus: str, vocab_size: int) -> None:
        """Learn merges from corpus until vocab reaches vocab_size."""

        # Convert text into byte token IDs
        tokens = list(corpus.encode("utf-8"))

        next_token_id = 256

        while next_token_id < vocab_size:

            # Count adjacent pairs
            pair_counts = Counter()

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # Most common pair
            best_pair, best_count = pair_counts.most_common(1)[0]

            # Stop if nothing repeats
            if best_count < 2:
                break

            # Create merged token
            new_token_id = next_token_id
            next_token_id += 1

            self.merges[best_pair] = new_token_id

            # Store bytes for new token
            self.vocab[new_token_id] = (
                self.vocab[best_pair[0]] +
                self.vocab[best_pair[1]]
            )

            # Replace pair occurrences
            new_tokens = []

            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1 and
                    (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        """Apply learned merges to text."""

        # Start from raw bytes
        tokens = list(text.encode("utf-8"))

        # Apply merges in LEARNED ORDER
        for pair, new_token_id in self.merges.items():

            new_tokens = []

            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1 and
                    (tokens[i], tokens[i + 1]) == pair
                ):
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""

        byte_stream = b"".join(self.vocab[token_id] for token_id in ids)

        return byte_stream.decode("utf-8", errors="replace")
