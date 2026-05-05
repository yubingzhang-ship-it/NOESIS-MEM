"""
PersonaMem Integration Test - Week 1 Complete Test

Tests the full pipeline:
  PersonaExtractor -> PersonaProfile -> MultiCriteriaRetriever -> ConsistencyChecker

Author: PersonaMem Team
Date: 2026-04-10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from noesis_ii.core import (
    PersonaProfile, MemoryTrace, LongTermImpact,
    PersonaExtractor, OCEANScores,
    MultiCriteriaRetriever, RetrievalCriteria,
    ConsistencyChecker
)


def test_full_pipeline():
    """Test the complete PersonaMem pipeline"""
    print("=" * 60)
    print("PersonaMem Integration Test - Full Pipeline")
    print("=" * 60)
    
    # Step 1: Extract personality from text
    print("\n[Step 1] Extract personality from user input...")
    extractor = PersonaExtractor()
    
    user_inputs = [
        "I love exploring new ideas and creative solutions.",
        "I always plan ahead and organize my work carefully.",
        "I enjoy meeting new people and social activities."
    ]
    
    personalities = []
    for text in user_inputs:
        ocean = extractor.extract_ocean(text)
        personalities.append(ocean)
        print(f"  Text: {text[:40]}...")
        print(f"  -> O:{ocean.openness:.2f} C:{ocean.conscientiousness:.2f} "
              f"E:{ocean.extraversion:.2f} A:{ocean.agreeableness:.2f} N:{ocean.neuroticism:.2f}")
    
    # Step 2: Store as memory traces
    print("\n[Step 2] Store as memory traces...")
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_integration.db')
    profile = PersonaProfile(db_path=db_path)
    
    for i, (text, ocean) in enumerate(zip(user_inputs, personalities)):
        trace_id = profile.store_experience(
            experience=text,
            trace_type="user_input",
            intensity=0.7,
            context={"ocean": ocean.to_dict()}
        )
        print(f"  Stored trace {i+1}: id={trace_id}")
    
    # Step 3: Aggregate personality
    print("\n[Step 3] Aggregate personality profile...")
    # 从存储的痕迹中计算平均人格
    all_traces = profile.retrieve_by_conditions(["user_input"], top_k=10)
    if all_traces:
        avg_openness = sum(t.get('context', {}).get('ocean', {}).get('openness', 0.5) for t in all_traces) / len(all_traces)
        avg_conscientiousness = sum(t.get('context', {}).get('ocean', {}).get('conscientiousness', 0.5) for t in all_traces) / len(all_traces)
        avg_extraversion = sum(t.get('context', {}).get('ocean', {}).get('extraversion', 0.5) for t in all_traces) / len(all_traces)
        avg_agreeableness = sum(t.get('context', {}).get('ocean', {}).get('agreeableness', 0.5) for t in all_traces) / len(all_traces)
        avg_neuroticism = sum(t.get('context', {}).get('ocean', {}).get('neuroticism', 0.5) for t in all_traces) / len(all_traces)
    else:
        avg_openness = avg_conscientiousness = avg_extraversion = avg_agreeableness = avg_neuroticism = 0.5
    
    aggregated = OCEANScores(
        openness=avg_openness,
        conscientiousness=avg_conscientiousness,
        extraversion=avg_extraversion,
        agreeableness=avg_agreeableness,
        neuroticism=avg_neuroticism
    )
    print(f"  Aggregated OCEAN:")
    print(f"    Openness: {aggregated.openness:.2f}")
    print(f"    Conscientiousness: {aggregated.conscientiousness:.2f}")
    print(f"    Extraversion: {aggregated.extraversion:.2f}")
    print(f"    Agreeableness: {aggregated.agreeableness:.2f}")
    print(f"    Neuroticism: {aggregated.neuroticism:.2f}")
    
    # Step 4: Check consistency
    print("\n[Step 4] Check consistency with new input...")
    checker = ConsistencyChecker()
    
    test_text = "I'm excited to try this new approach with the team!"
    report = checker.check_consistency(
        generated_text=test_text,
        target_persona=aggregated.to_dict(),
        extractor=extractor
    )
    
    print(f"  Test text: {test_text}")
    print(f"  Is consistent: {report.is_consistent}")
    print(f"  KL divergence: {report.kl_divergence:.4f}")
    print(f"  Drift score: {report.drift_score:.4f}")
    if report.suggestions:
        print(f"  Suggestions: {report.suggestions}")
    
    # Step 5: Retrieve relevant memories
    print("\n[Step 5] Retrieve relevant memories...")
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'noesis.db')
    
    if os.path.exists(db_path):
        retriever = MultiCriteriaRetriever(db_path)
        criteria = RetrievalCriteria(
            semantic_query="creative exploration",
            access_preference='recent'
        )
        results = retriever.retrieve(criteria, top_k=3)
        
        print(f"  Retrieved {len(results)} memories:")
        for i, result in enumerate(results, 1):
            content_preview = result.content[:50] if result.content else "N/A"
            print(f"    {i}. [ID:{result.memory_id}] Score:{result.score:.4f}")
            print(f"       Content: {content_preview}...")
    else:
        print(f"  [SKIP] Database not found: {db_path}")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully!")
    print("=" * 60)
    return True


def test_backward_compatibility():
    """Test backward compatibility with old module names"""
    print("\n" + "=" * 60)
    print("Backward Compatibility Test")
    print("=" * 60)
    
    try:
        from noesis_ii.core import AlayaSeeds, PersonalitySeeds, IABEngine
        print("  [OK] Deprecated modules can be imported (with warnings)")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("PersonaMem Week 1 - Complete Integration Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    try:
        if not test_full_pipeline():
            all_passed = False
    except Exception as e:
        print(f"\n[FAIL] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not test_backward_compatibility():
            all_passed = False
    except Exception as e:
        print(f"\n[FAIL] Backward compatibility test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
