from openai import OpenAI
from datasets import load_dataset
import backoff
from dotenv import dotenv_values
import pandas as pd


secrets = dotenv_values(".env")
client_openai = OpenAI(api_key=secrets["OPENAI_API_KEY"])
client_deepseek = OpenAI(
    api_key=secrets["DeepSeek_API_Key"], base_url="https://api.deepseek.com"
)
model_openai = "o1"
model_deepseek = "deepseek-reasoner"


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def get_completion(prompt: str, model: str = "deepseek-reasoner") -> str:
    """
    Get completion from OpenAI, or DeepSeek API with exponential backoff retry

    Args:
        prompt: The prompt text to send to the model
        model: Model identifier (o1, or deepseek-reasoner)

    Returns:
        str: The model's response text

    Raises:
        ValueError: If invalid model specified
        Exception: For API errors after max retries
    """
    try:
        if model == "deepseek-reasoner":
            response = client_deepseek.chat.completions.create(
                model=model_deepseek,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        elif model == "o1":
            response = client_openai.chat.completions.create(
                model=model_openai,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Invalid model specified: {model}")

    except Exception as e:
        print(f"Error during {model} API call: {str(e)}")
        raise


# Store results from model comparisons
results = []

# Load mathematical questions dataset and convert to list
dataset = list(load_dataset("mlfoundations-dev/seed_math_deepmind", split="train"))

# Process last 100 questions from dataset
for item in dataset[-200:]:
    print(item["question"])

    # Construct prompt with question and example format
    prompt = f"""
    
    Review the following question and provide a correct solution.
    Your response must end with a line in the exact form:
    Answer: <ANSWER>
    No additional commentary or text should appear after the final "Answer: <ANSWER>" line.

    Question: {item["question"]}

    Below are the five examples showing the exact format for the question and the final answer:
    
    Example 1: 
    Question: b"Suppose 8*l + 42 = 122. Find the third derivative of l*t**4 + 3*t**4 + t**4 - 4*t**4 - 6*t**2 wrt t.\n"
    Answer: b'240*t'

    Example 2: 
    Question: b"Let h(r) be the second derivative of -r**11/5040 - r**7/140 - r**4/12 - 20*r. Let j(z) be the third derivative of h(z). Find the third derivative of j(x) wrt x.\n"
    Answer: b'-1320*x**'
    
    Example 3:
    Question:  b'Let n(h) be the third derivative of 25*h**8/48 - h**5/60 + 31*h**4/12 + 189*h**2. Find the third derivative of n(b) wrt b.\n'
    Answer: b'10500*b**2'
    
    Example 4:
    Question: b'Let b be 4/8*-4*-1. Let l be ((-3)/b)/((-6)/20). Differentiate -8*x - 16 + 2*x - l*x + 2 with respect to x.\n'
    Answer: b'-11'
    
    Example 5:
    Question: b'Let l(n) be the second derivative of -48*n**5/5 + 62*n**4/3 - n**3/6 - 128*n + 1. What is the third derivative of l(p) wrt p?\n'
    Answer: b'-1152'

    """

    try:
        # Get responses from both models
        r1_answer = get_completion(prompt=prompt, model=model_deepseek)
        o1_answer = get_completion(prompt=prompt, model=model_openai)

        # Store results with cleaned answers
        results.append(
            {
                "question": item["question"],
                "correct_answer": item["answer"].strip("b'\\n'").replace(" ", ""),
                "deepseek_answer": r1_answer.split("Answer: ")[-1]
                .strip()
                .strip("b'")
                .strip("'")
                .replace(" ", ""),
                "o1_answer": o1_answer.split("Answer: ")[-1]
                .strip()
                .strip("b'")
                .strip("'")
                .replace(" ", ""),
            }
        )
    except:
        # Skip failed questions
        continue


# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("model_comparison_results.csv", index=False)
