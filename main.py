
import argparse

from code.procurement_graph import ask

def main():
    print("Let me get the information...")

    parser = argparse.ArgumentParser(
        description="Procurement chatbot"
    )

    parser.add_argument(
        "question",
        help="Procurement question to answer."
    )

    parser.add_argument(
        "--thread-id",
        default="default",
        help="Conversation identifier."
    )

    args = parser.parse_args()

    answer = ask(
        args.question,
        args.thread_id
    )

    print(answer)


if __name__ == "__main__":
    main()
