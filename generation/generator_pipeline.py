import json
import logging
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Generator, Optional, Any

import openai
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from textstat import textstat
from tqdm.auto import tqdm

# ============================================ 
# Configuration & Constants
# ============================================ 

class Config:
    MODEL_REPO = 'unsloth/Qwen3-32B-GGUF'
    MODEL_FILENAME = "Qwen3-32B-Q4_1.gguf"
    
    # Text Generation Parameters
    CTX_SIZE = 16384
    N_THREADS = 32
    N_GPU_LAYERS = -1
    BATCH_SIZE = 4096
    SEED = 3407
    FLASH_ATTN = True
    
    # Generation Settings
    MAX_TOKENS = 2048
    TEMPERATURE = 1.0
    TOP_P = 0.95
    TOP_K = 20
    
    # Pipeline Settings
    SAMPLES_PER_DOMAIN = 100
    SIMILARITY_THRESHOLD = 0.75
    MAX_RETRIES = 5
    
    DOMAINS = [
        "science", "technology", "health", "environment",
        "education", "economics", "politics", "history",
        "literature", "arts", "sports", "culture"
    ]

    CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

    CEFR_GUIDELINES = {
        "A1": {
            "description": "Basic user - Beginner",
            "vocabulary_description": "Common, concrete words for everyday things, people, and basic actions",
            "vocabulary_examples": "family, house, school, go, eat, water, big, happy",
            "phrase_examples": "'What's your name?', 'How are you?', 'Thank you.'",
            "sentence_structure": "Very short, simple sentences (5-8 words)",
            "grammar": "Present Simple, basic pronouns (I, you, he)",
            "example": "The cat is big. It is black. The cat sits on the mat."
        },
        "A2": {
            "description": "Basic user - Elementary",
            "vocabulary_description": "Words for personal information, daily routines, and direct exchange of information",
            "vocabulary_examples": "because, often, friendly, market, travel, decide, explain",
            "phrase_examples": "'What do you think?', 'I think that...', 'Can I help you?'",
            "sentence_structure": "Short sentences linked with basic conjunctions like 'and', 'but', 'because' (8-12 words)",
            "grammar": "Past Simple (regular/irregular verbs), 'going to' future",
            "example": "My cat is very friendly. Yesterday, it played in the garden. It will sleep soon."
        },
        "B1": {
            "description": "Independent user - Intermediate",
            "vocabulary_description": "Words to express opinions, feelings, and describe experiences and ambitions",
            "vocabulary_examples": "opinion, environment, government, probably, achieve, advantage",
            "phrase_examples": "'In my opinion...', 'On the one hand...', 'If I were you, I would...'",
            "sentence_structure": "Moderate sentences with simple clauses (12-15 words)",
            "grammar": "Present Perfect, conditionals (if-clauses), comparatives",
            "example": "I have been taking care of my cat for three years. If the weather is nice, we usually spend time outdoors together."
        },
        "B2": {
            "description": "Independent user - Upper intermediate",
            "vocabulary_description": "More abstract and precise words for discussing complex ideas, issues, and arguments",
            "vocabulary_examples": "consequence, responsible, evidence, significant, debate, complex",
            "phrase_examples": "'That's a valid point, but...', 'I see what you mean, however...'",
            "sentence_structure": "Complex sentences using relative clauses (15-20 words)",
            "grammar": "Passive voice, reported speech, advanced conditionals",
            "example": "Cats are known for their independent nature, which has made them popular pets throughout history. Studies have shown that interaction with cats can reduce stress levels."
        },
        "C1": {
            "description": "Proficient user - Advanced",
            "vocabulary_description": "Formal, academic, and idiomatic vocabulary to express complex or subtle ideas precisely",
            "vocabulary_examples": "nuanced, sophisticated, comprehensive, articulate, underlying",
            "phrase_examples": "'I'm inclined to disagree...', 'Having said that...', 'It's crucial that we consider...'",
            "sentence_structure": "Long, complex sentences with varied structures (20-25 words)",
            "grammar": "Inversion, complex passive structures, idiomatic language",
            "example": "Feline domestication, which began approximately 9,500 years ago, has resulted in numerous behavioral adaptations that distinguish domestic cats from their wild ancestors, particularly in their capacity for inter-species communication."
        },
        "C2": {
            "description": "Proficient user - Mastery",
            "vocabulary_description": "Sophisticated, specialized, and academic vocabulary to convey fine shades of meaning",
            "vocabulary_examples": "phylogenetic, repertoire, paradigm, malleable, ubiquitous",
            "phrase_examples": "'Be that as it may...', 'The crux of the matter is...'",
            "sentence_structure": "Very complex, sophisticated, and varied sentence structures (25+ words)",
            "grammar": "Mastery of idiomatic expressions, nuanced grammar, and register",
            "example": "The phylogenetic trajectory of Felis catus demonstrates remarkable neuroplasticity, manifesting in sophisticated cognitive architectures that facilitate both intra-specific and anthropocentric social paradigms, thereby exemplifying the profound malleability inherent in mammalian behavioral repertoires."
        }
    }

# ============================================ 
# Logger Setup
# ============================================ 

def setup_logger(name: str = "cefr_generator", log_dir: str = "logs") -> logging.Logger:
    """로거 설정 - 파일과 콘솔에 모두 기록"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        f"{log_dir}/cefr_generation_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')

    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

# ============================================ 
# 1. Text Generator
# ============================================ 

class ThinkTagError(Exception):
    pass

class TextGenerator:
    def __init__(self, model_repo: str = Config.MODEL_REPO, model_filename: str = Config.MODEL_FILENAME):
        logger.info(f"Downloading {model_repo}...")
        try:
            model_path = hf_hub_download(
                repo_id=model_repo,
                filename=model_filename,
            )
            logger.info(f"Model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

        logger.info("Loading model...")
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=Config.CTX_SIZE,
                n_threads=Config.N_THREADS,
                n_gpu_layers=Config.N_GPU_LAYERS,
                n_batch=Config.BATCH_SIZE,
                seed=Config.SEED,
                flash_attn=Config.FLASH_ATTN,
                verbose=False
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_text(self, prompt: str) -> Optional[str]:
        retry_count = 0
        backoff_time = 1
        
        while retry_count < Config.MAX_RETRIES:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    top_k=Config.TOP_K
                )
                content = response['choices'][0]['message']['content']
                return self._remove_think(content)
                
            except ThinkTagError:
                retry_count += 1
                logger.warning(f"Think tag error (Attempt {retry_count}/{Config.MAX_RETRIES})")
            except Exception as e:
                retry_count += 1
                logger.error(f"Generation error: {e} (Attempt {retry_count}/{Config.MAX_RETRIES})")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff

        logger.error("Max retries reached. Returning None.")
        return None

    @staticmethod
    def _remove_think(output_text: str) -> str:
        if '<think>' in output_text:
            if '</think>' not in output_text:
                raise ThinkTagError("Closing </think> tag missing")
        return re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

# ============================================ 
# 2. Sample Data Generator
# ============================================ 

class SampleGenerator:
    """SLM을 사용하여 샘플 텍스트와 질문 자동 생성 (배치 처리)"""

    def __init__(self, generator: TextGenerator):
        self.generator = generator
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        self.max_retries = Config.MAX_RETRIES
        self.embedding_model = self._load_embedding_model()

    def _load_embedding_model(self):
        logger.info("Loading semantic similarity model...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', token=False)
            logger.info("Semantic model loaded successfully")
            return model
        except ImportError:
            traceback.print_exc()
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None

    def _get_source_text_prompt(self, domain: str, min_words: int, max_words: int, existing_samples: List[str], max_history: int = 50) -> str:
        context = ""
        if existing_samples:
            context = "**Previously generated samples (DO NOT repeat these):**\n"
            for idx, sample in enumerate(existing_samples[-max_history:]):
                context += f"{idx + 1}. {sample}\n"

        return f"""Generate an educational, factual statement about {domain}.

                {context}**Requirements:**
                1. Write ONE complete sentence (or 2 short sentences)
                2. Length: {min_words}-{max_words} words
                3. Content: factual, neutral, suitable for English learners
                4. Style: clear and informative
                5. No questions, no opinions, just facts
                6. MUST be different from any previously generated samples above
                7. Final answer must be in plain text format
                
                **Output ONLY the statement, nothing else.**
                
                Statement:"""

    def _get_query_prompt(self, domain: str, min_words: int, max_words: int, existing_samples: List[str], max_history: int = 50) -> str:
        context = ""
        if existing_samples:
            context = "**Previously generated samples (DO NOT repeat these):**\n"
            for idx, sample in enumerate(existing_samples[-max_history:]):
                context += f"{idx + 1}. {sample}\n"

        return f"""Generate a simple question about {domain} that an English learner might ask.

                {context}**Requirements:**
                1. Write ONE clear question
                2. Length: {min_words}-{max_words} words
                3. Style: direct, simple, educational
                4. Should be answerable in 2-4 sentences
                5. Examples: "What is photosynthesis?", "How does the Internet work?"
                6. MUST ask about a different topic than previously generated questions above
                7. Final answer must be in plain text format
                
                **Output ONLY the question, nothing else.**
                
                Question:"""

    def generate_source_texts(self, num_per_domain: int = 2, min_words: int = 12, max_words: int = 25) -> List[Tuple[str, str]]:
        logger.info(f"\nGenerating source texts: {num_per_domain} per domain")
        samples = []
        domain_samples = {domain: [] for domain in Config.DOMAINS}

        total_needed = num_per_domain * len(Config.DOMAINS)
        
        with tqdm(total=total_needed, desc="Generating Source Texts") as pbar:
            while True:
                domains_need_more = [d for d in Config.DOMAINS if len(domain_samples[d]) < num_per_domain]
                if not domains_need_more:
                    break

                for domain in domains_need_more:
                    prompt = self._get_source_text_prompt(domain, min_words, max_words, domain_samples[domain])
                    response = self.generator.generate_text(prompt)
                    
                    if not response: continue

                    candidate_text = self._clean_sample(response)

                    if not self._validate_text(candidate_text, min_words, max_words):
                        continue

                    is_dup, _ = self._semantic_duplicate_check(candidate_text, domain_samples[domain])
                    if is_dup:
                        continue

                    domain_samples[domain].append(candidate_text)
                    samples.append((candidate_text, domain))
                    pbar.update(1)
                    logger.debug(f"[{domain}] Generated: {candidate_text}")

        return samples

    def generate_queries(self, num_per_domain: int = 2, min_words: int = 5, max_words: int = 12) -> List[Tuple[str, str]]:
        logger.info(f"\nGenerating queries: {num_per_domain} per domain")
        samples = []
        domain_samples = {domain: [] for domain in Config.DOMAINS}

        total_needed = num_per_domain * len(Config.DOMAINS)

        with tqdm(total=total_needed, desc="Generating Queries") as pbar:
            while True:
                domains_need_more = [d for d in Config.DOMAINS if len(domain_samples[d]) < num_per_domain]
                if not domains_need_more:
                    break

                for domain in domains_need_more:
                    prompt = self._get_query_prompt(domain, min_words, max_words, domain_samples[domain])
                    response = self.generator.generate_text(prompt)

                    if not response: continue

                    candidate_query = self._clean_sample(response)

                    if not self._validate_query(candidate_query):
                        continue

                    is_dup, _ = self._semantic_duplicate_check(candidate_query, domain_samples[domain])
                    if is_dup:
                        continue

                    domain_samples[domain].append(candidate_query)
                    samples.append((candidate_query, domain))
                    pbar.update(1)
                    logger.debug(f"[{domain}] Generated: {candidate_query}")

        return samples

    @staticmethod
    def _clean_sample(text: str) -> str:
        text = text.strip().strip('"\'')
        prefixes = ["Statement:", "Question:", "Answer:", "Text:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            text = sentences[0]
            if not text.endswith(('.', '!', '?')):
                text += '.'
        return text.strip()

    def _semantic_duplicate_check(self, candidate: str, existing_samples: List[str]) -> Tuple[bool, float]:
        if not existing_samples or self.embedding_model is None:
            return False, 0.0

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        candidate_embedding = self.embedding_model.encode(candidate, convert_to_tensor=False)
        existing_embeddings = self.embedding_model.encode(existing_samples, convert_to_tensor=False)

        candidate_embedding = candidate_embedding.reshape(1, -1)
        similarities = cosine_similarity(candidate_embedding, existing_embeddings)[0]
        max_similarity = float(np.max(similarities))

        return max_similarity >= self.similarity_threshold, max_similarity

    @staticmethod
    def _validate_text(text: str, min_words: int, max_words: int) -> bool:
        if not text or len(text) < 10: return False
        words = text.split()
        if len(words) < min_words or len(words) > max_words + 10: return False
        if text.strip().endswith('?'): return False
        return True

    @staticmethod
    def _validate_query(query: str) -> bool:
        if not query or len(query) < 5: return False
        words = query.split()
        if len(words) < 3 or len(words) > 20: return False
        
        query_lower = query.lower()
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'does', 'is', 'are']
        starts_with_question = any(query_lower.startswith(qw) for qw in question_words)
        ends_with_question = query.strip().endswith('?')
        return starts_with_question or ends_with_question

# ============================================ 
# 3. Parallel Corpus Generator
# ============================================ 

class ParallelCorpusGenerator:
    """CEFR 레벨별 parallel corpus 생성"""

    def _get_prompt(self, level: str, guidelines: Dict[str, str], source_text: str) -> str:
        return f"""Rewrite the following text for a CEFR {level} English learner.

                **CEFR {level} Guidelines:**
                - **Vocabulary Type:** Use {guidelines['vocabulary_description']}. For example: {guidelines['vocabulary_examples']}.
                - **Functional Phrases:** Where appropriate, use phrases like: {guidelines['phrase_examples']}.
                - **Sentence Structure:** Aim for {guidelines['sentence_structure']}.
                - **Grammar:** Use grammar appropriate for this level, such as: {guidelines['grammar']}.
                
                **Original text:**
                {source_text}
                
                **CRITICAL REQUIREMENTS:**
                1. Strictly adhere to the {level} guidelines.
                2. Preserve the core meaning of the original text.
                3. Use the specified type of vocabulary and phrases naturally.
                4. Output ONLY the rewritten text. Do not add any explanations.
                
                **Rewritten text at CEFR {level}:**"""

    def generate_parallel_versions(self, text_generator: TextGenerator, sources: List[Tuple[str, str]]) -> Generator[Dict, None, None]:
        """Generator that yields parallel data items one by one."""
        
        for source_text, topic in tqdm(sources, desc="Generating parallel versions"):
            version_responses = {}
            for level in Config.CEFR_LEVELS:
                guidelines = Config.CEFR_GUIDELINES[level]
                prompt = self._get_prompt(level, guidelines, source_text)
                response = text_generator.generate_text(prompt)
                
                # If generation fails, we might skip this level or retry. 
                # For now, if response is None, we store an empty string or handle it.
                version_responses[level] = response if response else ""

            parallel_data = {
                "id": f"parallel_{abs(hash(source_text)) % 1_000_0000}",
                "topic": topic,
                "source_text": source_text,
                "versions": {},
                "validated": False
            }

            for level, response in version_responses.items():
                if response:
                    metrics = self._calculate_metrics(response, level)
                    parallel_data["versions"][level] = {"text": response, **metrics}
                else:
                     parallel_data["versions"][level] = {"text": "", "error": "Generation failed"}

            yield parallel_data

    def _calculate_metrics(self, text: str, level: str) -> Dict:
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Safe division
        avg_sentence_len = len(words) / max(len(sentences), 1)
        
        try:
            return {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_sentence_length": avg_sentence_len,
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "dale_chall": textstat.dale_chall_readability_score(text),
                "difficult_words_count": textstat.difficult_words(text),
                "difficult_words": textstat.difficult_words_list(text),
                "level": level
            }
        except Exception:
            # Fallback for empty text or textstat errors
            return {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "level": level,
                "error": "metric_calc_failed"
            }

# ============================================ 
# 4. Instruction Data Generator
# ============================================ 

class InstructionDataGenerator:
    """CEFR 레벨별 instruction-response 쌍 생성"""

    def _get_prompt(self, level: str, guidelines: Dict[str, str], query: str) -> str:
        return f"""You are a helpful AI assistant. Your user is an English learner at CEFR level {level}.
                Your goal is to answer their question accurately while adapting your language to their level.
                
                **User's Query:** {query}
                
                **Your Response Guidelines (CEFR {level}):**
                - **Vocabulary Type:** Use {guidelines['vocabulary_description']}. For example: {guidelines['vocabulary_examples']}.
                - **Functional Phrases:** Be conversational using phrases like: {guidelines['phrase_examples']}.
                - **Sentence Structure:** Keep sentences clear: {guidelines['sentence_structure']}.
                - **Grammar:** Use grammar they can understand, such as: {guidelines['grammar']}.
                
                **CRITICAL REQUIREMENTS:**
                1. Answer the user's query helpfully and accurately.
                2. Strictly use language appropriate for a {level} learner.
                3. Be natural and conversational.
                4. Provide complete information in simple, easy-to-understand terms.
                
                **Your response:**"""

    def generate_instruction_pair(self, text_generator: TextGenerator, sources: List[Tuple[str, str]]) -> Generator[Dict, None, None]:
        """Generator that yields instruction data items one by one."""
        
        for query, domain in tqdm(sources, desc="Generating Instructions"):
            for level in Config.CEFR_LEVELS:
                guidelines = Config.CEFR_GUIDELINES[level]
                prompt = self._get_prompt(level, guidelines, query)
                response = text_generator.generate_text(prompt)
                
                if response:
                    yield {
                        "id": f"instruction_{abs(hash(query + level)) % 1_000_0000}",
                        "user_level": level,
                        "conversation": [
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": response, "level_used": level}
                        ],
                        "domain": domain
                    }

# ============================================ 
# 5. Main Execution
# ============================================ 

def main():
    logger.info("=" * 60)
    logger.info("CEFR Dataset Generation Pipeline")
    logger.info("=" * 60)

    # Initialize Generator
    try:
        text_generator = TextGenerator()
    except Exception:
        logger.critical("Failed to initialize TextGenerator. Exiting.")
        return

    # 1. Generate Sample Data
    logger.info("\nStep 1: Generating Sample Data...")
    sample_gen = SampleGenerator(text_generator)

    sample_texts = sample_gen.generate_source_texts(num_per_domain=Config.SAMPLES_PER_DOMAIN)
    sample_queries = sample_gen.generate_queries(num_per_domain=Config.SAMPLES_PER_DOMAIN)

    # Save Intermediate Samples (Optional but good for safety)
    Path("data/intermediate").mkdir(parents=True, exist_ok=True)
    with open("data/intermediate/source_texts.json", "w", encoding="utf-8") as f:
        json.dump(sample_texts, f, indent=2, ensure_ascii=False)
    with open("data/intermediate/queries.json", "w", encoding="utf-8") as f:
        json.dump(sample_queries, f, indent=2, ensure_ascii=False)

    # 2. Generate Parallel Corpus (Incremental Save)
    output_path = Path("../data/parallel_corpus/train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nStep 2: Generating Parallel Corpus -> {output_path}")
    
    parallel_gen = ParallelCorpusGenerator()
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in parallel_gen.generate_parallel_versions(text_generator, sample_texts):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush() # Ensure data is written to disk
            count += 1
            
    logger.info(f"Generated {count} parallel corpus entries.")

    # 3. Generate Instruction Data (Incremental Save)
    output_path = Path("../data/instructions/train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nStep 3: Generating Instruction Data -> {output_path}")
    
    instruction_gen = InstructionDataGenerator()
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in instruction_gen.generate_instruction_pair(text_generator, sample_queries):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            count += 1

    logger.info(f"Generated {count} instruction entries.")
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Completed Successfully!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
