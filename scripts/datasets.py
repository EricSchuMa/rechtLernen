import os

import pandas as pd


def extract_qa_pairs(text: str) -> list[dict[str, str]]:
    lines = text.split("\n")  # Split the entire text into lines
    qa_pairs = []
    current_question = None
    current_answer = []
    expected_question_number = 1

    for line in lines:
        # Check if the line starts with the expected question number followed by a dot
        if line.startswith(f"{expected_question_number}."):
            # If there is a current question, save it along with its accumulated answer
            if current_question is not None:
                qa_pairs.append({"question": current_question, "answer": " ".join(current_answer).strip()})
            # Update the current question and reset the answer
            current_question = line.strip()
            current_answer = []
            expected_question_number += 1
        else:
            # Accumulate lines to the current answer
            current_answer.append(line)

    # Don't forget to add the last question and answer pair
    if current_question is not None:
        qa_pairs.append({"question": current_question, "answer": " ".join(current_answer).strip()})

    return qa_pairs


def load_qa_pairs_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return extract_qa_pairs(text)


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
    )
    stgb_test = os.path.join(data_path, "StGB-Klausur.txt")
    qa_dataset = load_qa_pairs_from_file(stgb_test)
    # Save to CSV
    df = pd.DataFrame(qa_dataset)
    df.to_csv(os.path.join(data_path, "StGB-QA.csv"), index=False)
