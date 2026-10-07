import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from src.graph.workflow import build_graph

def run_evaluation():
    with open("evaluation/golden_dataset.json", "r") as f:
        golden_cases = json.load(f)
        
    app = build_graph()
    questions, answers, contexts, ground_truths = [], [], [], []
    
    for case in golden_cases:
        res = app.invoke({"question": case["question"]})
        questions.append(case["question"])
        answers.append(res.get("final_answer", ""))
        contexts.append([c["text"] for c in res.get("retrieved_contexts", [])] or ["Consulta SQL estruturada"])
        ground_truths.append(case["ground_truth"])
        
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    
    print("\n--- Resultados de Avaliação RAGAS ---")
    print(results)
    
    # Salva relatório de métricas em json
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()