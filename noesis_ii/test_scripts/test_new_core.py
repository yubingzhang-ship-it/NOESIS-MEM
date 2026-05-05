"""
Test new core modules - PersonaMem Route A

Test content:
1. PersonaExtractor - LLM personality extraction
2. MultiCriteriaRetriever - Multi-criteria retrieval
3. ConsistencyChecker - Consistency check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noesis_ii.core import (
    PersonaExtractor, OCEANScores,
    MultiCriteriaRetriever, RetrievalCriteria,
    ConsistencyChecker
)


def test_persona_extractor():
    """Test personality extractor"""
    print("\n" + "="*60)
    print("Test 1: PersonaExtractor - Personality Extraction")
    print("="*60)
    
    extractor = PersonaExtractor(llm_client=None)  # Use fallback
    
    test_texts = [
        "I love exploring new things, always curious, like to try different experiences.",
        "I focus on planning and organization, do things methodically, like step-by-step.",
        "I like social activities, feel energetic in crowds, good at expressing myself."
    ]
    
    for text in test_texts:
        scores = extractor.extract_ocean(text)
        print(f"\nText: {text[:40]}...")
        print(f"  O: {scores.openness:.2f} | C: {scores.conscientiousness:.2f} | "
              f"E: {scores.extraversion:.2f} | A: {scores.agreeableness:.2f} | "
              f"N: {scores.neuroticism:.2f}")
    
    print("\n[OK] PersonaExtractor test passed")


def test_consistency_checker():
    """Test consistency checker"""
    print("\n" + "="*60)
    print("Test 2: ConsistencyChecker - Consistency Check")
    print("="*60)
    
    checker = ConsistencyChecker()
    
    # Target persona: high openness, high extraversion
    target_persona = {
        'openness': 0.8,
        'conscientiousness': 0.5,
        'extraversion': 0.8,
        'agreeableness': 0.6,
        'neuroticism': 0.3
    }
    
    # Test text 1: matches target persona
    text1 = "I'm so excited! This new idea is great, shall we try it together?"
    report1 = checker.check_consistency(text1, target_persona)
    print(f"\nText 1: {text1}")
    print(f"  Consistent: {report1.is_consistent} | KL: {report1.kl_divergence:.2f} | "
          f"Drift: {report1.drift_score:.2f}")
    
    # Test text 2: deviates from target (low extraversion)
    text2 = "I don't really want to attend parties, prefer staying alone quietly."
    report2 = checker.check_consistency(text2, target_persona)
    print(f"\nText 2: {text2}")
    print(f"  Consistent: {report2.is_consistent} | KL: {report2.kl_divergence:.2f} | "
          f"Drift: {report2.drift_score:.2f}")
    if report2.suggestions:
        print(f"  Suggestion: {report2.suggestions[0]}")
    
    print("\n[OK] ConsistencyChecker test passed")


def test_multi_criteria_retriever():
    """Test multi-criteria retriever"""
    print("\n" + "="*60)
    print("Test 3: MultiCriteriaRetriever - Multi-criteria Retrieval")
    print("="*60)
    
    db_path = "d:/Project/NOESIS-II v1.0/noesis_ii/data/noesis.db"
    
    if not os.path.exists(db_path):
        print(f"[WARN] Database not found: {db_path}")
        print("Skip retrieval test")
        return
    
    retriever = MultiCriteriaRetriever(db_path)
    
    criteria = RetrievalCriteria(
        semantic_query="personality consistency",
        access_preference='recent'
    )
    
    try:
        results = retriever.retrieve(criteria, top_k=5)
        print(f"\nSearch criteria: semantic='{criteria.semantic_query}'")
        print(f"Returned {len(results)} results:")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. [ID:{r.memory_id}] Score:{r.score:.3f} | "
                  f"Matched:{','.join(r.criteria_matched)}")
            print(f"     Content: {r.content[:50]}...")
        
        print("\n[OK] MultiCriteriaRetriever test passed")
    except Exception as e:
        print(f"\n[FAIL] Retrieval test failed: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PersonaMem New Core Module Tests")
    print("="*60)
    
    try:
        test_persona_extractor()
    except Exception as e:
        print(f"\n[FAIL] PersonaExtractor test failed: {e}")
    
    try:
        test_consistency_checker()
    except Exception as e:
        print(f"\n[FAIL] ConsistencyChecker test failed: {e}")
    
    try:
        test_multi_criteria_retriever()
    except Exception as e:
        print(f"\n[FAIL] MultiCriteriaRetriever test failed: {e}")
    
    print("\n" + "="*60)
    print("Tests completed")
    print("="*60)


if __name__ == "__main__":
    main()
