import time
import json
from rag_engine import instantiate_llm, load_and_embed_defaults, generate_answer

# Load Vectorstore
print("Loading vector store...")
vectorstore, files, num_chunks = load_and_embed_defaults()
print(f"Loaded {num_chunks} chunks from {files}")

# Instantiate both LLMs
print("Initializing LLMs...")
llm_local = instantiate_llm("Local (Ollama - Mistral)")
llm_cloud = instantiate_llm("OpenRouter")

questions = [
    "What is Indecimal's one-line summary?",
    "Are the package pricing rates inclusive of GST?",
    "What is the Escrow-Based Payment Model?",
    "What kind of home construction does Indecimal support?",
    "Who built the Eiffel Tower?",
    "What are the 3 tiers of protection policies?",
    "How does the system ensure quality?",
    "Can I build a 50-story commercial skyscraper with Indecimal?",
    "Who is the CEO of Indecimal?",
    "How many critical checkpoints are in the quality assurance system?"
]

results = []

print("Starting evaluation (this might take a few minutes)...")

for i, q in enumerate(questions):
    print(f"\n--- Q{i+1}: {q} ---")
    
    # 1. Local Evaluation
    start_local = time.time()
    try:
        ans_local, _ = generate_answer(q, vectorstore, llm_local, [])
    except Exception as e:
        ans_local = f"ERROR: {str(e)}"
    time_local = time.time() - start_local
    print(f"Local time: {time_local:.2f}s")
    
    # 2. Cloud Evaluation
    start_cloud = time.time()
    try:
        ans_cloud, _ = generate_answer(q, vectorstore, llm_cloud, [])
    except Exception as e:
        ans_cloud = f"ERROR: {str(e)}"
    time_cloud = time.time() - start_cloud
    print(f"Cloud time: {time_cloud:.2f}s")
    
    results.append({
        "question": q,
        "local_answer": ans_local,
        "local_time_seconds": time_local,
        "cloud_answer": ans_cloud,
        "cloud_time_seconds": time_cloud
    })

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nEvaluation complete! Results saved to eval_results.json")
