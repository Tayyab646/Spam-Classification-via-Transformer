import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from PIL import Image, ImageTk
import numpy as np
import re
import os
import json

class GlassSpamClassifier:
    def __init__(self):
        # Initialize paths for saving models
        self.model_dir = "saved_models"
        self.tokenizer_dir = os.path.join(self.model_dir, "tokenizer")
        self.model_path = os.path.join(self.model_dir, "model")
        self.model_info_path = os.path.join(self.model_dir, "model_info.json")
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_model()
        
        # Set up customtkinter appearance - Light theme for glass effect
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("🌈 Glass Spam Classifier AI")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.create_glass_gui()
    
    def model_exists(self):
        """Check if model is already saved locally"""
        return (os.path.exists(self.tokenizer_dir) and 
                os.path.exists(self.model_path) and
                len(os.listdir(self.model_path)) > 0 and
                len(os.listdir(self.tokenizer_dir)) > 0)
    
    def save_model_info(self):
        """Save model information to JSON file"""
        model_info = {
            "model_name": "mariagrandury/roberta-base-finetuned-sms-spam-detection",
            "saved_date": pd.Timestamp.now().isoformat(),
            "model_type": "RoBERTa for SMS Spam Detection",
            "version": "1.0"
        }
        with open(self.model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def load_model_info(self):
        """Load model information from JSON file"""
        if os.path.exists(self.model_info_path):
            with open(self.model_info_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_model(self):
        """Load model from local storage or download if not exists"""
        try:
            if self.model_exists():
                print("📂 Loading model from local storage...")
                model_info = self.load_model_info()
                if model_info:
                    print(f"✅ Model loaded: {model_info['model_name']}")
                    print(f"📅 Saved on: {model_info['saved_date']}")
                
                # Load tokenizer and model from local files
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.model.eval()
                print("🎯 Model loaded successfully from local storage!")
                
            else:
                print("🌐 Downloading model from Hugging Face...")
                # Download and save the model
                model_name = "mariagrandury/roberta-base-finetuned-sms-spam-detection"
                
                # Download tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Save them locally
                print("💾 Saving model to local storage...")
                self.tokenizer.save_pretrained(self.tokenizer_dir)
                self.model.save_pretrained(self.model_path)
                self.save_model_info()
                
                self.model.eval()
                print("✅ Model downloaded and saved successfully!")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try fallback model
            try:
                print("🔄 Trying fallback model...")
                model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                print("✅ Fallback model loaded successfully!")
            except Exception as e2:
                messagebox.showerror("Error", f"Failed to load any model: {e2}")
    
    def contains_spam_patterns(self, message):
        """Rule-based spam detection for common patterns"""
        message_lower = message.lower()
        
        spam_indicators = [
            # Financial spam patterns
            r'\b(won|winner|winning|prize|reward|cash|money|million|billion|dollar|€|£|₹)\b',
            r'\b(free|gift|card|voucher|coupon|bonus|reward)\b',
            r'\b(claim|collect|call now|text now|reply now|act now|urgent)\b',
            r'\b(urgent|important|alert|warning|attention|immediate)\b',
            r'\b(lottery|raffle|jackpot|bonanza)\b',
            r'\b(selected|chosen|winner|lucky|congratulation)\b',
            
            # URL and contact patterns
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'\b(bit\.ly|tinyurl|goo\.gl|t\.co|shorte\.st|adf\.ly|url\.zip)\b',
            r'\b(\d{10,13})\b',
            r'\b[A-Z0-9]{10,}\b',
            
            # Spammy phrases
            r'\b(limited time|special promotion|exclusive offer|no cost)\b',
            r'\b(guaranteed|risk-free|no obligation|no purchase necessary)\b',
            r'\b(text stop to end|reply stop to unsubscribe|opt out)\b',
            
            # Suspicious financial offers
            r'\b(loan|credit|debt|mortgage|insurance|investment)\b',
            r'\b(earn|income|salary|profit|make money|work from home)\b',
        ]
        
        spam_score = 0
        for pattern in spam_indicators:
            matches = re.findall(pattern, message_lower)
            if matches:
                spam_score += len(matches)
        
        # Check for excessive capitalization
        if len(message) > 10:
            upper_ratio = sum(1 for c in message if c.isupper()) / len(message)
            if upper_ratio > 0.5:
                spam_score += 2
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in message if not c.isalnum() and not c.isspace()) / len(message)
        if special_char_ratio > 0.3:
            spam_score += 1
        
        return spam_score >= 2
    
    def create_glass_gui(self):
        """Create beautiful glassomorphic GUI with compatible colors"""
        
        # Color palette for glass effect
        self.colors = {
            "bg_light": "#f8f9fa",
            "bg_lighter": "#ffffff",
            "primary": "#667eea",
            "primary_dark": "#5a6fd8",
            "secondary": "#764ba2",
            "success": "#2ecc71",
            "success_dark": "#27ae60",
            "danger": "#e74c3c",
            "danger_dark": "#c0392b",
            "warning": "#f39c12",
            "info": "#3498db",
            "light_gray": "#e9ecef",
            "text_dark": "#2c3e50",
            "text_light": "#6c757d",
            "white": "#ffffff"
        }
        
        # Main container with glass effect
        main_frame = ctk.CTkFrame(
            self.root,
            corner_radius=25,
            border_width=0,
            fg_color=self.colors["bg_light"],
            bg_color=self.colors["bg_light"]
        )
        main_frame.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Set window background
        self.root.configure(fg_color=self.colors["bg_light"])
        
        # Title section with gradient background
        title_frame = ctk.CTkFrame(
            main_frame,
            corner_radius=20,
            fg_color=self.colors["primary"],
            border_width=0,
            height=120
        )
        title_frame.grid(row=0, column=0, padx=25, pady=(25, 20), sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🔮Spam Classifier",
            font=ctk.CTkFont(size=32, weight="bold", family="Arial"),
            text_color=self.colors["white"],
            bg_color="transparent"
        )
        title_label.grid(row=0, column=0, pady=(20, 5))
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="AI-Powered SMS Spam Detection ",
            font=ctk.CTkFont(size=14, family="Arial"),
            text_color=self.colors["white"],
            bg_color="transparent"
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 20))
        
        # Model status badge
        model_status = "📂 Local Model" if self.model_exists() else "🌐 Online Model"
        status_color = self.colors["success"] if self.model_exists() else self.colors["warning"]
        
        status_frame = ctk.CTkFrame(
            title_frame,
            corner_radius=15,
            fg_color=status_color,
            border_width=0,
            height=30
        )
        status_frame.grid(row=2, column=0, pady=(0, 15))
        
        status_label = ctk.CTkLabel(
            status_frame,
            text=model_status,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors["white"],
            bg_color="transparent"
        )
        status_label.pack(padx=15, pady=5)
        
        # Input section with glass effect
        input_frame = ctk.CTkFrame(
            main_frame,
            corner_radius=20,
            fg_color=self.colors["bg_lighter"],
            border_color=self.colors["light_gray"],
            border_width=2
        )
        input_frame.grid(row=1, column=0, padx=25, pady=15, sticky="ew")
        
        input_label = ctk.CTkLabel(
            input_frame,
            text="💌 Enter SMS Message:",
            font=ctk.CTkFont(size=18, weight="bold", family="Arial"),
            text_color=self.colors["text_dark"]
        )
        input_label.pack(pady=(25, 15))
        
        # Text area with glass effect
        self.text_area = ctk.CTkTextbox(
            input_frame,
            height=130,
            corner_radius=15,
            border_width=2,
            border_color=self.colors["light_gray"],
            fg_color=self.colors["bg_lighter"],
            text_color=self.colors["text_dark"],
            font=ctk.CTkFont(size=14, family="Arial"),
            scrollbar_button_color=self.colors["info"],
            scrollbar_button_hover_color=self.colors["primary"]
        )
        self.text_area.pack(padx=25, pady=(0, 20), fill="x")
        self.text_area.insert("1.0", "Type your SMS message here...")
        
        # Button section
        button_frame = ctk.CTkFrame(
            input_frame,
            fg_color="transparent",
            bg_color="transparent"
        )
        button_frame.pack(pady=15)
        
        # Analyze button with gradient effect
        self.analyze_btn = ctk.CTkButton(
            button_frame,
            text="🎯 Analyze Message",
            command=self.analyze_message,
            font=ctk.CTkFont(size=16, weight="bold", family="Arial"),
            height=45,
            corner_radius=12,
            fg_color=self.colors["info"],
            hover_color=self.colors["primary"],
            border_width=0,
            text_color=self.colors["white"]
        )
        self.analyze_btn.pack(side="left", padx=10)
        
        # Clear button
        clear_btn = ctk.CTkButton(
            button_frame,
            text="✨ Clear",
            command=self.clear_text,
            font=ctk.CTkFont(size=14, family="Arial"),
            height=45,
            corner_radius=12,
            fg_color=self.colors["warning"],
            hover_color="#e67e22",
            border_width=0,
            text_color=self.colors["white"]
        )
        clear_btn.pack(side="left", padx=10)
        
        # Test buttons frame
        test_frame = ctk.CTkFrame(
            input_frame,
            fg_color="transparent",
            bg_color="transparent"
        )
        test_frame.pack(pady=10)
        
        test_spam_btn = ctk.CTkButton(
            test_frame,
            text="🧪 Test Spam Example",
            command=self.test_spam_example,
            font=ctk.CTkFont(size=12, family="Arial"),
            height=35,
            corner_radius=10,
            fg_color=self.colors["danger"],
            hover_color=self.colors["danger_dark"],
            border_width=0,
            text_color=self.colors["white"]
        )
        test_spam_btn.pack(side="left", padx=5)
        
        test_ham_btn = ctk.CTkButton(
            test_frame,
            text="🧪 Test Legit Example",
            command=self.test_ham_example,
            font=ctk.CTkFont(size=12, family="Arial"),
            height=35,
            corner_radius=10,
            fg_color=self.colors["success"],
            hover_color=self.colors["success_dark"],
            border_width=0,
            text_color=self.colors["white"]
        )
        test_ham_btn.pack(side="left", padx=5)
        
        # Results section (initially hidden)
        self.result_frame = ctk.CTkFrame(
            main_frame,
            corner_radius=20,
            fg_color=self.colors["bg_lighter"],
            border_color=self.colors["light_gray"],
            border_width=2
        )
        self.result_frame.grid(row=2, column=0, padx=25, pady=(0, 25), sticky="ew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_remove()
        
        # Stats section with glass effect
        stats_frame = ctk.CTkFrame(
            main_frame,
            corner_radius=15,
            fg_color=self.colors["primary"],
            border_width=0
        )
        stats_frame.grid(row=3, column=0, padx=25, pady=(0, 25), sticky="ew")
        
        model_info = self.load_model_info() or {}
        saved_date = model_info.get('saved_date', 'Not available')
        
        stats_text = f"""
✨ AI Model Information:
• 🤖 Model: RoBERTa Base Fine-tuned
• 💾 Storage: {'Local' if self.model_exists() else 'Online'}
• 📅 Saved: {saved_date[:10] if saved_date != 'Not available' else 'N/A'}
• 🎯 Detection: Hybrid AI + Rules
• 🌈 UI: Glass Morphism Design
        """
        
        stats_label = ctk.CTkLabel(
            stats_frame,
            text=stats_text,
            font=ctk.CTkFont(size=12, family="Consolas"),
            text_color=self.colors["white"],
            justify="left",
            bg_color="transparent"
        )
        stats_label.pack(padx=20, pady=15)
    
    def test_spam_example(self):
        """Load a test spam example"""
        spam_example = "🎉 CONGRATULATIONS! You've won a $1000 Walmart gift card! Click here to claim your prize: http://bit.ly/win-now Limited time offer! 🎊"
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", spam_example)
    
    def test_ham_example(self):
        """Load a test legitimate example"""
        ham_example = "Hey, are we still meeting for lunch tomorrow at 1 PM? Let me know if that works for you. 😊"
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", ham_example)
    
    def analyze_message(self):
        """Enhanced analysis with hybrid detection"""
        message = self.text_area.get("1.0", "end-1c").strip()
        
        if not message or message == "Type your SMS message here...":
            messagebox.showwarning("Input Required", "Please enter an SMS message to analyze.")
            return
        
        # First check with rule-based system
        rule_based_spam = self.contains_spam_patterns(message)
        
        if rule_based_spam:
            # If rule-based says spam, show immediate result
            self.show_results(True, 0.95, 0.95, message, rule_based=True)
            return
        
        # Otherwise, use AI model for ambiguous cases
        self.use_ai_detection(message)
    
    def use_ai_detection(self, message):
        """Use AI model for detection"""
        if not self.model or not self.tokenizer:
            messagebox.showerror("Error", "AI model is not loaded. Please try again.")
            return
        
        try:
            self.analyze_btn.configure(text="🔄 Analyzing...", state="disabled")
            self.root.update()
            
            # Tokenize the message
            inputs = self.tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Handle different model output formats
            if predictions.shape[1] == 2:  # Binary classification
                ham_prob = predictions[0][0].item()
                spam_prob = predictions[0][1].item()
            else:
                spam_prob = predictions[0][-1].item()
                ham_prob = 1 - spam_prob
            
            # Use higher threshold for spam detection
            is_spam = spam_prob > 0.7
            confidence = spam_prob if is_spam else ham_prob
            
            self.show_results(is_spam, confidence, spam_prob, message, rule_based=False)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"AI Analysis failed: {str(e)}")
        finally:
            self.analyze_btn.configure(text="🎯 Analyze Message", state="normal")
    
    def show_results(self, is_spam, confidence, spam_prob, message, rule_based=False):
        """Display beautiful glass results"""
        self.result_frame.grid()
        
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Result colors and icons
        if rule_based:
            result_text = "🚨 SPAM DETECTED! "
            result_color = self.colors["danger"]
            result_icon = "🔍"
            detection_type = "Pattern Matching System"
            confidence_text = "High Confidence (Rule-Based)"
        else:
            if is_spam:
                result_text = "🚨 SPAM DETECTED! (AI)"
                result_color = self.colors["danger"]
                result_icon = "🤖"
            else:
                result_text = "✅ LEGITIMATE MESSAGE"
                result_color = self.colors["success"]
                result_icon = "💚"
            detection_type = "AI Neural Network"
            confidence_text = f"AI Confidence: {confidence*100:.1f}%"
        
        # Result header
        result_header = ctk.CTkLabel(
            self.result_frame,
            text=f"{result_icon} {result_text}",
            font=ctk.CTkFont(size=22, weight="bold", family="Arial"),
            text_color=result_color,
            bg_color="transparent"
        )
        result_header.grid(row=0, column=0, pady=(25, 10))
        
        # Create a frame for the result content
        content_frame = ctk.CTkFrame(
            self.result_frame,
            corner_radius=15,
            fg_color=self.colors["light_gray"],
            border_width=0
        )
        content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Detection type
        type_label = ctk.CTkLabel(
            content_frame,
            text=f"🔮 {detection_type}",
            font=ctk.CTkFont(size=14, family="Arial"),
            text_color=self.colors["text_dark"],
            bg_color="transparent"
        )
        type_label.grid(row=0, column=0, pady=(15, 5))
        
        # Confidence information
        confidence_label = ctk.CTkLabel(
            content_frame,
            text=f"📊 {confidence_text}",
            font=ctk.CTkFont(size=16, weight="bold", family="Arial"),
            text_color=self.colors["text_dark"],
            bg_color="transparent"
        )
        confidence_label.grid(row=1, column=0, pady=5)
        
        if not rule_based:
            # Progress bar for AI results
            progress_bar = ctk.CTkProgressBar(
                content_frame,
                width=300,
                height=20,
                corner_radius=10,
                progress_color=result_color,
                fg_color=self.colors["bg_lighter"],
                border_width=0
            )
            progress_bar.grid(row=2, column=0, pady=10)
            progress_bar.set(confidence)
            
            # Detailed probabilities
            prob_text = f"🎯 Spam: {spam_prob*100:.1f}% | 💚 Legit: {(1-spam_prob)*100:.1f}%"
            prob_label = ctk.CTkLabel(
                content_frame,
                text=prob_text,
                font=ctk.CTkFont(size=13, family="Arial"),
                text_color=self.colors["text_dark"],
                bg_color="transparent"
            )
            prob_label.grid(row=3, column=0, pady=(5, 15))
        
        # Message preview with glass effect
        preview_frame = ctk.CTkFrame(
            self.result_frame,
            corner_radius=15,
            fg_color=self.colors["bg_lighter"],
            border_color=self.colors["light_gray"],
            border_width=2
        )
        preview_frame.grid(row=2, column=0, padx=20, pady=15, sticky="ew")
        
        preview_label = ctk.CTkLabel(
            preview_frame,
            text="💌 Message Analyzed:",
            font=ctk.CTkFont(size=14, weight="bold", family="Arial"),
            text_color=self.colors["text_dark"]
        )
        preview_label.pack(pady=(15, 10))
        
        message_preview = ctk.CTkLabel(
            preview_frame,
            text=f'"{message}"',
            font=ctk.CTkFont(size=12, family="Arial"),
            text_color=self.colors["text_light"],
            wraplength=600,
            justify="center"
        )
        message_preview.pack(pady=(0, 15), padx=20)
    
    def clear_text(self):
        """Clear the text area and hide results"""
        self.text_area.delete("1.0", "end")
        self.result_frame.grid_remove()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function to run the application"""
    print("🚀 Starting Glass Spam Classifier...")
    print("📂 Checking for local model...")
    print("🎨 Loading beautiful glass interface...")
    
    app = GlassSpamClassifier()
    app.run()

if __name__ == "__main__":
    main()