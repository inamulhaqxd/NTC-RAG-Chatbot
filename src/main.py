from rag import answer_question


while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    answer = answer_question(question)

    print("\nAnswer:")
    print(answer)