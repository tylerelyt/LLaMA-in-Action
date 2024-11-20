# Lesson 5: Basic Prompting Techniques

This lesson introduces fundamental prompting techniques for interacting with Large Language Models. You will learn the difference between zero-shot and few-shot prompting and see how to apply them in practice.

## Key Learning Objectives

- **Few-Shot Prompting**: Understand how to provide examples within a prompt to guide the model's output.
- **Zero-Shot Prompting**: Learn how to ask a model to perform a task without providing any prior examples.

## File Descriptions

- `example1.py`: Demonstrates few-shot prompting for a sentiment analysis task. The prompt includes positive and negative examples to guide the model.
- `example2.py`: Shows a zero-shot prompt where the model is asked to explain a complex topic without any guiding examples.
- `requirements.txt`: Lists all the necessary Python dependencies for this lesson.

## Setup and Execution

1.  **Install Dependencies**:
    Install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Examples**:
    Execute the individual scripts to see the prompting techniques in action.
    ```bash
    # Few-shot prompting
    python example1.py

    # Zero-shot prompting
    python example2.py
    ``` 