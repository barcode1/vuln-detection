import torch
import numpy as np
from src.models.classifier import CodeBERTVulnClassifier
from src.data_pipeline.tokenizer import MultiEmbeddingTokenizer
import yaml
import logging


class ManualTester:
    def __init__(self, config_path: str = '../config/hyperparams.yaml', model_path: str = 'best_model.pth'):
        """بارگذاری مدل برای تست دستی"""
        print("🔄 Loading model for manual testing...")

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize tokenizer
        self.tokenizer = MultiEmbeddingTokenizer(self.config)
        self.tokenizer.load_word_embeddings()

        # Initialize model
        self.model = CodeBERTVulnClassifier(self.config)

        # Load checkpoint
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.classifier.load_state_dict(checkpoint['model_state_dict'])

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        # Class names
        self.class_names = ['SQLi', 'XSS', 'CommandInjection', 'Normal']

        print("✅ Model loaded successfully!")
        print(f"📱 Device: {self.device}")

    def predict(self, text: str):
        """پیش‌بینی برای یک متن"""
        print(f"\n{'=' * 70}")
        print(f"🔍 Input: {text[:100]}...")
        print('=' * 70)

        # Encode input (same as training)
        encoded = self.tokenizer.encode([text])

        # Move to device
        input_ids = encoded['sec_bert']['input_ids'].to(self.device)
        attention_mask = encoded['sec_bert']['attention_mask'].to(self.device)
        word2vec = encoded['word2vec'].to(self.device)
        fasttext = encoded['fasttext'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model.classifier(
                input_ids, attention_mask, word2vec, fasttext
            )

            # Classifier results
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            classifier_pred = torch.argmax(probs, dim=1).item()

            # Anomaly detection
            features = outputs['features']
            anomaly_score = self.model.anomaly_detector.detect_anomaly(features)
            is_anomaly = anomaly_score > self.config['anomaly_detection']['anomaly_threshold']

            # Ensemble decision
            fusion_weight = self.config['fusion']['weights'][0]  # classifier weight
            ensemble_prob = (probs[0, classifier_pred] * fusion_weight +
                             (1 - anomaly_score) * (1 - fusion_weight))

        # Display results
        print("\n📊 Classifier Results:")
        print("─" * 40)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:15s}: {probs[0, i]:.4f}")

        print(f"\n🎯 Primary Prediction: {self.class_names[classifier_pred]}")
        print(f"   Confidence: {probs[0, classifier_pred]:.4f}")

        print(f"\n🔍 Anomaly Detection:")
        print("─" * 40)
        print(f"Anomaly Score: {anomaly_score:.4f}")
        print(f"Is Anomaly: {'Yes ⚠️' if is_anomaly else 'No ✅'}")

        print(f"\n⚖️  Ensemble Decision:")
        print("─" * 40)
        print(f"Final Score: {ensemble_prob:.4f}")

        final_decision = (self.class_names[classifier_pred]
                          if not is_anomaly else "ANOMALY ⚠️")
        print(f"Final Label: {final_decision}")

        return {
            'probs': probs.cpu().numpy(),
            'prediction': classifier_pred,
            'prediction_name': self.class_names[classifier_pred],
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'ensemble_prob': ensemble_prob,
            'final_decision': final_decision
        }

    def interactive_test(self):
        """حلقه تست دستی"""
        print("\n" + "=" * 70)
        print("🧪 Manual Testing Mode (Ctrl+C to exit)")
        print("=" * 70)

        test_cases = [
            "SELECT * FROM users WHERE id = 1",
            "<script>alert('XSS')</script>",
            "rm -rf /var/www/html",
            "Hello this is normal text",
            "admin' OR '1'='1",
            "<img src=x onerror=alert(1)>",
            "DROP TABLE students;--",
            "Normal API request with parameters"
        ]

        print("\n📋 Sample test cases:")
        for i, case in enumerate(test_cases, 1):
            print(f"{i}. {case[:60]}...")

        while True:
            try:
                print("\n" + "-" * 70)
                print("Choose option:")
                print("1. Use sample test case")
                print("2. Enter custom text")
                print("3. Exit")
                choice = input("> ").strip()

                if choice == '1':
                    print("\nEnter test case number (1-8):")
                    num = int(input("> ").strip())
                    if 1 <= num <= len(test_cases):
                        text = test_cases[num - 1]
                    else:
                        print("❌ Invalid number!")
                        continue

                elif choice == '2':
                    print("\nEnter your text:")
                    text = input("> ").strip()

                elif choice == '3':
                    print("👋 Exiting...")
                    break
                else:
                    print("❌ Invalid choice!")
                    continue

                # Predict
                if text:
                    self.predict(text)

            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize tester
    tester = ManualTester(
        config_path='../config/hyperparams.yaml',
        model_path='best_model.pth'
    )

    # Start interactive testing
    tester.interactive_test()