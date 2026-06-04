import argparse
import json
from pathlib import Path

from rag import KeywordRetriever, build_rag_context


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RAG context from a traffic violation pipeline JSON result."
    )
    parser.add_argument(
        "pipeline_result",
        help="Path to JSON output produced by main.py.",
    )
    parser.add_argument(
        "--knowledge-base",
        default="rag/knowledge_base.json",
        help="Path to legal knowledge base JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of legal documents to retrieve.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the RAG context JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.pipeline_result, "r", encoding="utf-8") as file:
        pipeline_result = json.load(file)

    retriever = KeywordRetriever(args.knowledge_base)
    rag_context = build_rag_context(
        pipeline_result=pipeline_result,
        retriever=retriever,
        top_k=args.top_k,
    )
    output = json.dumps(rag_context, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()
