
import re
import html
import unicodedata


def clean_text(text):
    """
    Clean news text before TF-IDF vectorization.

    Steps:
    1. Handle missing values
    2. Convert HTML entities
    3. Normalize Unicode
    4. Convert to lowercase
    5. Remove URLs
    6. Remove HTML tags
    7. Remove email addresses
    8. Remove non-alphabetic characters
    9. Normalize whitespace
    """

    if text is None:
        return ""

    text = str(text)

    # HTML entities
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Email addresses
    text = re.sub(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text 