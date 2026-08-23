from src.graph import ask


def main() -> None:
    print("Multimodal RAG Knowledge Assistant — type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = ask(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
