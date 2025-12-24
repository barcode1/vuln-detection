import re
import urllib.parse
import logging
from typing import List, Dict, Any


class SecurityPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def fit_transform(self, texts: List[str]) -> List[str]:
        processed = []
        for text in texts:
            try:
                # 1. URL Decoding
                if self.config.get('url_decode', True):
                    text = urllib.parse.unquote_plus(text)

                #delet commeents
                text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
                text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
                text = re.sub(r'\s+', ' ', text).strip()


                text = self._preserve_malicious_patterns(text)

                processed.append(text)
            except Exception as e:
                self.logger.error(f"Preprocessing error: {e}")
                processed.append("")

        return processed

    def _preserve_malicious_patterns(self, text: str) -> str:
        # SQL keywords
        text = re.sub(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR|AND)\b',
                      lambda m: m.group(1), text, flags=re.IGNORECASE)

        # XSS patterns
        text = re.sub(r'(<script|javascript:|on\w+=|<img|alert\(|document\.cookie)',
                      lambda m: m.group(1), text, flags=re.IGNORECASE)

        # Command injection
        text = re.sub(r'([;&|`$()]{1,2})', lambda m: m.group(1), text)

        return text