import re
from typing import List, Tuple, Dict, Optional
import spacy
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate

class QueryDecomposerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self._nlp = None

        # Optimization: a keyword library classified by intent, with different weights
        self.INTENT_PATTERNS = {
            "comparison": {
                "keywords": ["difference", "diff", "different", "vs", "compare", "pros and cons", "which is better", "comparison"],
                "weight": 1.2
            },
            "multi_hop": {
                "keywords": ["first.*then.*", "workflow", "steps", "how to implement", "principle", "how to do", "how to deploy", "how.*"],
                "weight": 1.0
            },
            "aggregation": {
                "keywords": ["summarize", "generalize", "common points", "total"],
                "weight": 0.5
            }
        }

        # Entity type weight configuration
        self.ENTITY_TYPE_WEIGHTS = {
            "ORG": 1.0,      # Organization
            "PERSON": 0.9,   # Person name
            "GPE": 0.8,      # Geopolitical entity (place name)
            "PRODUCT": 0.7,  # Product
            "EVENT": 0.6,    # Event
            "WORK_OF_ART": 0.5,  # Work of art
            "LAW": 0.5,      # Law
            "LANGUAGE": 0.4, # Language
            "DATE": 0.3,     # Date
            "TIME": 0.3,     # Time
            "PERCENT": 0.2,  # Percentage
            "MONEY": 0.2,    # Money
            "QUANTITY": 0.2, # Quantity
            "CARDINAL": 0.1, # Cardinal
            "ORDINAL": 0.1   # Ordinal
        }

        # Part-of-speech weight configuration
        self.POS_WEIGHTS = {
            "NOUN": 0.8,     # Noun
            "PROPN": 0.9,    # Proper noun
            "VERB": 0.6,     # Verb
            "ADJ": 0.4,      # Adjective
            "ADV": 0.3,      # Adverb
        }

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("zh_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "The spaCy Chinese model is not installed. Please run: python -m spacy download zh_core_web_sm"
                )
        return self._nlp

    # Stopword filtering
    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Filter using a complete stopword list.
        """
        # Expanded stopword list
        stop_words = {
            "what", "how", "why", "the", "a", "an", "and", "or", "of", "in", "on", "at", "by", "to", "for", "with",
            "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would",
            "can", "could", "should", "shall", "may", "might", "must", "not", "no", "yes", "so", "very", "too", "just",
            "also", "then", "now", "here", "there", "this", "that", "these", "those", "it", "its", "they", "them",
            "about", "above", "after", "again", "against", "all", "am", "any", "as", "before", "below", "between",
            "but", "from", "into", "off", "over", "under", "until", "up", "down", "out", "if", "while", "such", "like",
            "furthermore", "moreover", "however", "although", "though", "because", "since", "therefore", "thus",
            "besides", "except", "either", "nor", "each", "every", "own", "other", "another", "one", "two", "first",
            "second", "last", "next", "finally", "already", "still", "yet", "again", "once", "never", "always",
            "often", "sometimes", "usually", "perhaps", "maybe", "really", "actually", "only", "than"
        }
        return [token for token in tokens if token not in stop_words]

    # Entity normalization
    def _normalize_entities(self, entities: List[str]) -> List[str]:
        """
        Entity normalization: deduplicate, filter empty values, remove single characters.
        """
        seen = set()
        normalized = []
        for entity in entities:
            entity = entity.strip()
            if len(entity) > 1 and entity not in seen:
                seen.add(entity)
                normalized.append(entity)
        return normalized

    # Get entity type weight
    def _get_entity_weight(self, entity_type: str) -> float:
        """
        Get the weight based on the entity type.
        """
        return self.ENTITY_TYPE_WEIGHTS.get(entity_type, 0.3)

    @lru_cache(maxsize=256)
    def _extract_entities_nlp(self, query: str) -> List[str]:
        """
        Use spaCy for intelligent entity extraction, including NER and keyword extraction.
        Use an LRU cache to optimize performance.
        """
        # Use spaCy for entity recognition
        doc = self.nlp(query)

        # 1. Extract named entities (NER), storing the doc from the previous step into named_entities
        named_entities = []
        for ent in doc.ents:
            named_entities.append(ent.text)

        # 2. Extract keywords based on part-of-speech tagging
        keywords = []
        for token in doc:
            # Extract nouns, proper nouns, and verbs
            if token.pos_ in ["NOUN", "PROPN", "VERB"]:
                # Filter stopwords and single characters
                if len(token.text) > 1 and not token.is_stop:
                    keywords.append(token.text)

        # 3. Merge and normalize entities
        all_entities = named_entities + keywords
        filtered_entities = self._filter_stopwords(all_entities)
        normalized_entities = self._normalize_entities(filtered_entities)

        return normalized_entities

    @lru_cache(maxsize=256)
    def _calculate_complexity_score(self, query: str, entities_tuple: tuple) -> float:
        """
        Calculate the complexity score, considering entity type, part of speech, keywords, and syntactic structure.
        Use an LRU cache to optimize performance; entities_tuple is used for cache compatibility.
        """
        score = 0.0
        doc = self.nlp(query)
        entities = list(entities_tuple)

        # 1. Basic entity count evaluation - distinguish whether it is a true multi-entity query that needs decomposition
        if len(entities) >= 2:
            # Check whether a real comparison or relation keyword exists
            has_comparison = any(kw in query for intent_data in self.INTENT_PATTERNS.values()
                                for kw in intent_data["keywords"] if intent_data.get("weight", 0) >= 0.8)
            # Check whether it is an attribute query pattern, such as "population of Beijing", "market cap of Apple", etc.
            has_attribute_pattern = any([
                re.search(r'(.*)(population|market cap|stock price|GDP|housing price|area|length|weight|capacity|sales volume|revenue|profit|price|cost)', query),
                re.search(r'(.*)(founder|author|inventor|developer|creator)', query),
                re.search(r'(.*)(history|development|origin|background|introduction|definition|concept)', query)
            ])
            if has_comparison:
                score += 1.0  # Comparison query, needs decomposition
            elif has_attribute_pattern:
                score += 0.1  # Attribute query, no decomposition needed
            else:
                # Multiple entities but neither comparison nor attribute query; possibly a parallel query
                score += 0.3  # Moderate bonus, but still needs LLM judgment
        elif len(entities) == 0:
            score -= 0.2

        # 2. Weighting by named entity type
        entity_types_found = set()
        for ent in doc.ents:
            entity_weight = self._get_entity_weight(ent.label_)
            score += entity_weight * 0.3
            entity_types_found.add(ent.label_)

        # 3. Bonus for combinations of multiple entity types
        if len(entity_types_found) >= 2:
            score += 0.2

        # 4. Weighting by part-of-speech combination
        pos_tags = [token.pos_ for token in doc]
        if "NOUN" in pos_tags and "VERB" in pos_tags:
            score += 0.15
        if "PROPN" in pos_tags:
            score += 0.1

        # 5. Keyword weighting - more precise intent recognition
        for intent, data in self.INTENT_PATTERNS.items():
            for kw in data["keywords"]:
                if re.search(kw, query) if '.' in kw else kw in query:
                    score += data["weight"]
                    # One hit is enough; avoid duplicate bonus
                    break

        # 6. Syntactic structure weighting - distinguish connectives that truly require decomposition
        # Specifically for comparison connectives
        if re.search(r'.*(difference|diff|different|compare|comparison|vs|which is better).*and.*', query):
            score += 0.6  # Clear comparison intent
        elif re.search(r'.+and.+', query):
            # Non-comparative "and" connection, e.g. "population of Beijing and Shanghai" vs "difference between Beijing and Shanghai"
            if any(comp_word in query for comp_word in ["difference", "diff", "different", "compare", "comparison", "vs"]):
                score += 0.5  # Has comparison intent
            else:
                score += 0.1  # Pure parallel relationship

        if re.search(r'.+and.+', query):
            if any(comp_word in query for comp_word in ["difference", "diff", "different", "compare", "comparison", "vs"]):
                score += 0.5  # Has comparison intent
            else:
                score += 0.1  # Pure parallel relationship

        if re.search(r'.+or.+', query):
            score += 0.25

        # 7. Complex sentence pattern detection
        if re.search(r'.+first.+then.+', query):
            score += 1.0  # Clear multi-step intent
        # Detect "how" more precisely, excluding simple question words
        has_how_question = re.search(r'.+how.+', query) or (re.search(r'.*how.*', query) and 'how about' not in query)
        if has_how_question:
            # Check whether it is a simple question, e.g. "how to deploy" vs "how to deploy and configure"
            if any(word in query for word in ["workflow", "steps", "method", "how to do"]):
                score += 0.8  # Clear multi-step intent
            elif re.search(r'.+how.+(and).+', query) or re.search(r'.*how.*(and).*', query):  # e.g. "how to deploy and configure"
                score += 0.7  # Contains multiple operations
            elif re.search(r'.*how.*make.*', query) or re.search(r'.+how.+make.+', query):  # e.g. "how to make RAG search more accurately"
                score += 0.8  # Expresses an intent to improve a method; usually a complex query
            else:
                # For simple "how" queries, check whether multiple entities or a complex topic are involved
                if len(entities) <= 2:
                    score += 0.6  # "how" queries usually need decomposition even if there are few entities
                else:
                    score += 0.6  # A "how" query involving multiple entities may need decomposition
        # Handle simple interrogatives like "how about" and "what is"
        elif 'how about' in query or 'what is' in query or 'what does it look like' in query:
            # Judge whether it is a simple query
            if len(entities) <= 1:
                score -= 0.3  # Simple "how about" or "what is" query; reduce decomposition tendency

        return score

    def route_query(self, query: str) -> Tuple[bool, str, List[str]]:
        """
        Main entry point: decide the processing strategy.
        Returns: (needs_decomposition, reason, sub_questions)
        """

        # 1. Entity extraction, normalization, and deduplication
        entities = self._extract_entities_nlp(query)
        # Convert entities to a tuple to support caching
        score = self._calculate_complexity_score(query, tuple(entities))

        print(f"🔍 Query: {query} | Score: {score:.2f} | Entities: {entities}")

        # Threshold strategy - more conservative to reduce unnecessary decomposition
        HIGH_THRESHOLD = 0.75  # Confident that decomposition is needed
        LOW_THRESHOLD = 0.45  # Confident that it is not needed

        if score >= HIGH_THRESHOLD:
            print("🚀 Rule-based decomposition triggered")
            return True, "High Complexity Score", self._generate_subquestions_llm(query)

        elif score <= LOW_THRESHOLD:
            print("🛑 Simple query, retrieve directly")
            return False, "Simple Query", [query]

        else:
            print("⚖️ Ambiguous range, calling the LLM for a lightweight judgment")
            is_complex = self._llm_light_check(query)
            if is_complex:
                return True, "LLM Judged Complex", self._generate_subquestions_llm(query)
            else:
                return False, "LLM Judged Simple", [query]

    def _llm_light_check(self, query: str) -> bool:
        """
        Optimization point 3: Few-shot prompting to improve accuracy.
        """
        prompt = f"""## Task objective
        Accurately judge whether the user query contains **multiple intents** or requires **multi-step reasoning**, to decide whether query decomposition is needed.

        ---

        ## Judgment criteria (answer "yes" if any is met)

        **1. Multi-entity comparison**
        - Explicit comparison verbs: difference, diff, compare, comparison, vs, which is better, pros and cons
        - Implicit comparison: [attribute] of A and B (e.g. "population of Beijing and Shanghai")

        **2. Multi-step operation**
        - Process verbs: how, how to, steps, workflow, tutorial, guide
        - Temporal connectives: first...then..., then, next, finally

        **3. Aggregation analysis**
        - Summarize, generalize, common points, differences, comprehensive analysis

        **4. Conditional combination**
        - or, either...or (e.g. "Python or Java for backend development")

        ---

        ## Example library

        ### ❌ No decomposition needed (answer "no")
        - "What's the weather today" -> single-fact query
        - "Who is the founder of Python" -> single entity, single attribute
        - "Population of Beijing" -> single-entity attribute query
        - "What is deep learning" -> definitional question
        - "Recommend a sci-fi movie" -> subjective recommendation
        - "Latest version of ChatGPT" -> single-entity latest information

        ### ✅ Decomposition needed (answer "yes")
        - "Market cap comparison of Apple and Microsoft" -> dual-entity comparison
        - "How to deploy the DeepSeek model" -> multi-step workflow
        - "Difference between Transformer and CNN" -> technical comparison
        - "Which is better to live in, Beijing or Shanghai" -> conditional comparison
        - "Summarize AI breakthroughs in 2024" -> aggregation summary
        - "First install Python, then configure the environment variables" -> temporal steps

        ### ⚠️ Boundary cases (judge carefully)
        - "DeepSeek and ChatGPT" -> **yes** (missing attribute but implies comparison)
        - "Python data analysis and machine learning" -> **yes** (multiple topics)
        - "How to learn deep learning" -> **no** (single topic, no multi-step words)
        - "MySQL index optimization techniques" -> **no** (single-point technique)
        - "Population ranking of Beijing, Shanghai, and Guangzhou" -> **yes** (three-entity aggregation)

        ---

        ## Current query
        User input: "{query}"

        ## Thinking steps
        1. Identify the number of entities (>= 2 and not an attribute query?)
        2. Detect keywords (does it hit comparison/workflow/aggregation words?)
        3. Judge the sentence structure (does it contain "and/or/first...then"?)
        4. Exclude simple patterns (single entity + attribute/definition/recommendation)

        ## Output requirements
        Output only **yes** or **no**, with no explanation."""
        # Call llm.invoke ...
        # return True if "yes" in result else False
        return True # Mock

    def _generate_subquestions_llm(self, query: str) -> List[str]:
        """Sub-question generation function."""
        print(f"[Sub-question] Sub-question generation: {query}")

        try:
            subquestion_template = """User's original query: {query}
            Task: Split it into 2-5 independent sub-questions, satisfying:
            1. Each sub-question corresponds to only 1 information point, with no overlap;
            2. Preserve the core context of the original query (such as time, subject, scenario) without losing key constraints;
            3. The sub-questions can be used directly for retrieval (no additional information required);
            4. Do not generate redundant sub-questions (e.g. do not split "compare A and B" into "what is A" and "what is B").
            Output format: List the sub-questions numbered 1., 2., 3., etc., with no other content."""

            subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)
            response = self.llm.invoke(subquestion_prompt.format_messages(query=query))
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            # Parse the sub-question list
            sub_questions = []
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.\s*', line):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question:
                        sub_questions.append(question)

            print(f"  [Success] Generated {len(sub_questions)} sub-questions:")
            for i, q in enumerate(sub_questions, 1):
                print(f"    {i}. {q}")

            return sub_questions
        except Exception as e:
            print(f"  [Warning] Error generating sub-questions: {e}")
            # Return the default sub-question list
            return [f"Related information about {query}"]

# Usage suggestion
# agent = QueryDecomposerAgent(llm_client)
# needs_decomp, reason, qs = agent.route_query("Compare the inference cost of DeepSeek and ChatGPT")
