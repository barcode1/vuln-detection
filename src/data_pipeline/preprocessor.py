# import re
# import urllib.parse
# import logging
# from typing import List, Dict, Any
#
#
# class SecurityPreprocessor:
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
#
#     def fit_transform(self, texts: List[str]) -> List[str]:
#         processed = []
#         for text in texts:
#             try:
#                 # 1. URL Decoding
#                 if self.config.get('url_decode', True):
#                     text = urllib.parse.unquote_plus(text)
#
#                 #delet commeents
#                 text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
#                 text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
#                 text = re.sub(r'\s+', ' ', text).strip()
#
#
#                 text = self._preserve_malicious_patterns(text)
#
#                 processed.append(text)
#             except Exception as e:
#                 self.logger.error(f"Preprocessing error: {e}")
#                 processed.append("")
#
#         return processed
#
#     def _preserve_malicious_patterns(self, text: str) -> str:
#         # SQL keywords
#         text = re.sub(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR|AND)\b',
#                       lambda m: m.group(1), text, flags=re.IGNORECASE)
#
#         # XSS patterns
#         text = re.sub(r'(<script|javascript:|on\w+=|<img|alert\(|document\.cookie)',
#                       lambda m: m.group(1), text, flags=re.IGNORECASE)
#
#         # Command injection
#         text = re.sub(r'([;&|`$()]{1,2})', lambda m: m.group(1), text)
#
#         return text
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
        for i, text in enumerate(texts):
            try:
                # ذخیره نسخه اصلی برای logging
                original_sample = text[:100]

                # 1. URL Decoding (مهم برای تزریق‌ها)
                if self.config.get('url_decode', True):
                    text = urllib.parse.unquote_plus(text)

                # 2. حذف نظرات
                text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
                text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
                text = re.sub(r'\s+', ' ', text).strip()

                # 3. Decode Hex/Decimal (جدید!)
                if self.config.get('decode_obfuscation', True):
                    text = self._decode_hex_decimal(text)

                # 4. Preserve case-sensitive patterns (حفظ حروف بزرگ)
                text = self._preserve_malicious_patterns(text)

                # ✅ Logging در جای درست
                if self.config.get('verbose', False):
                    self.logger.info(f"Sample {i}: Original: {original_sample}...")
                    self.logger.info(f"Sample {i}: Processed: {text[:100]}...")

                processed.append(text)

            except Exception as e:
                self.logger.error(f"Preprocessing error at sample {i}: {e}")
                processed.append("")  # اضافه کردن متن خالی در صورت خطا

        return processed

    def _decode_hex_decimal(self, text: str) -> str:
        """
        Decode کردن عبارات hex و decimal رایج در حملات obfuscation
        """

        # Pattern 1: char(47,101,116,99) → "/etc"
        def decode_char(match):
            try:
                numbers = match.group(1).split(',')
                chars = [chr(int(n.strip())) for n in numbers if n.strip().isdigit()]
                return ''.join(chars)
            except:
                return match.group(0)

        text = re.sub(r'char\s*\(\s*([\d,]+)\s*\)', decode_char, text, flags=re.IGNORECASE)

        # Pattern 2: 0x2f657463 → "/etc"
        def decode_hex(match):
            try:
                hex_str = match.group(1)
                if len(hex_str) % 2 == 0:
                    return bytes.fromhex(hex_str).decode('utf-8', errors='ignore')
                return match.group(0)
            except:
                return match.group(0)

        text = re.sub(r'0x([0-9a-fA-F]+)', decode_hex, text)

        # Pattern 3: \x2f → "/"
        def decode_c_hex(match):
            try:
                hex_str = match.group(0)[2:]  # حذف \x
                return chr(int(hex_str, 16))
            except:
                return match.group(0)

        text = re.sub(r'\\x[0-9a-fA-F]{2}', decode_c_hex, text)

        return text

    def _preserve_malicious_patterns(self, text: str) -> str:
        """
        حفظ الگوهای مخرب بدون تغییر case
        """
        # SQL keywords - حفظ حروف بزرگ برای تشخیص بهتر
        sql_keywords = r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|OR|AND|FROM|TABLE|CREATE|ALTER|EXEC)\b'
        text = re.sub(sql_keywords, lambda m: m.group(1), text, flags=re.IGNORECASE)

        # XSS patterns - حفظ حساسیت به case
        xss_patterns = r'(<script|javascript:|on\w+=|<img|alert\(|document\.cookie|document\.write|eval\(|window\.location)'
        text = re.sub(xss_patterns, lambda m: m.group(1), text, flags=re.IGNORECASE)

        # Command injection - حفظ نمادها
        cmd_patterns = r'([;&|`$(){}[\]{}])'
        text = re.sub(cmd_patterns, lambda m: m.group(1), text)

        return text