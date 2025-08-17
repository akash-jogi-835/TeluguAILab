import re
import os
from typing import Optional

try:
    from llm_client import LLMClient
except Exception:
    LLMClient = None


class NLPProcessor:
    """Handles basic NLP processing (summarization, Q&A) with optional LLM backend"""

    def __init__(self, llm_client = None):
        self.llm = llm_client
        self.stop_words = {
            'english': [
                'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
                'is','are','was','were','be','been','being','have','has','had','do','does','did',
                'will','would','could','should','may','might','must','can','this','that','these','those'
            ],
            'telugu': ['ఇది','అది','ఈ','ఆ','వారు','మనం','మేము','నేను','నువ్వు','అతను','ఆమె']
        }

    # ---------------- LLM-enabled methods ----------------
    def summarize_text(self, text, model_name='basic', max_length=150):
        if model_name.lower().startswith("advanced") and self.llm is not None:
            return self.llm.summarize(text)

        # --- fallback: extractive summary ---
        cleaned_text = self.preprocess_text(text)
        if len(cleaned_text.strip()) < 10:
            return "Text is too short to summarize."

        sentences = self._extract_sentences(cleaned_text)
        if len(sentences) <= 2:
            return cleaned_text[:max_length] + "..." if len(cleaned_text) > max_length else cleaned_text

        words = re.findall(r'\b\w+\b', cleaned_text.lower())
        word_freq = {}
        for word in words:
            if word not in self.stop_words.get('english', []) and word not in self.stop_words.get('telugu', []):
                word_freq[word] = word_freq.get(word, 0) + 1

        sentence_scores = []
        for sentence in sentences:
            score = self._calculate_sentence_score(sentence, word_freq)
            sentence_scores.append((sentence, score))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        summary_sentences = []
        current_length = 0
        for sentence, score in sentence_scores:
            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
                if len(summary_sentences) >= 3:
                    break

        if not summary_sentences and sentence_scores:
            best_sentence = sentence_scores[0][0]
            summary_sentences = [best_sentence[:max_length] + "..."]

        return '. '.join(summary_sentences) + '.'

    def answer_question(self, context, question, model_name='basic'):
        if model_name.lower().startswith("advanced") and self.llm is not None:
            return self.llm.answer(context, question)

        cleaned_context = self.preprocess_text(context)
        if len(cleaned_context.strip()) < 10:
            return "Context too short."
        if len(question.strip()) < 3:
            return "Question too short."

        sentences = self._extract_sentences(cleaned_context)
        question_words = re.findall(r'\b\w+\b', question.lower())
        question_keywords = [
            w for w in question_words
            if w not in self.stop_words.get('english', [])
            and w not in self.stop_words.get('telugu', [])
            and len(w) > 2
        ]
        if not question_keywords:
            return "No keywords found in the question."

        sentence_scores = []
        for sentence in sentences:
            score = sum(1 for kw in question_keywords if kw in sentence.lower())
            if score > 0:
                sentence_scores.append((sentence, score))

        if sentence_scores:
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            return sentence_scores[0][0]
        else:
            return "No relevant information found."

    # ---------------- Helpers ----------------
    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'--- Page Break ---', '\n', text)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        return '\n'.join(cleaned_lines)

    def _extract_sentences(self, text):
        sentences = re.split(r'[.!?।]\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _calculate_sentence_score(self, sentence, word_freq):
        words = re.findall(r'\b\w+\b', sentence.lower())
        score = 0
        word_count = 0
        for word in words:
            if word not in self.stop_words.get('english', []) and word not in self.stop_words.get('telugu', []):
                score += word_freq.get(word, 0)
                word_count += 1
        return score / max(word_count, 1)
