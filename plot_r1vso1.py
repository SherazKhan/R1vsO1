import sympy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("model_comparison_results.csv").to_dict("records")


def symbolic_equal(prediction: str, ground_truth: str) -> bool:
    """
    Compare two expressions symbolically (using sympy).
    Returns True if they are equivalent, False otherwise.
    """
    # Replace '^' with '**' to handle caret notation
    pred_expr_str = prediction.replace("^", "**")
    gt_expr_str = ground_truth.replace("^", "**")

    try:
        pred_expr = sympy.sympify(pred_expr_str)
        gt_expr = sympy.sympify(gt_expr_str)
        # Check if their difference simplifies to 0
        return sympy.simplify(pred_expr - gt_expr) == 0
    except:
        # If parsing fails, fall back to direct string compare (strip whitespace)
        return pred_expr_str.strip() == gt_expr_str.strip()


###############################################################################
# Counters
deepseek_correct = 0
o1_correct = 0
total = len(data)

# Compare each item in data
for index, item in enumerate(data):
    correct_expr = item["correct_answer"]
    deepseek_expr = item["deepseek_answer"]
    o1_expr = item["o1_answer"]

    # Check deepseek correctness
    if symbolic_equal(deepseek_expr, correct_expr):
        deepseek_correct += 1
        data[index]["deepseek_correct"] = True
    else:
        data[index]["deepseek_correct"] = False

    # Check o1 correctness
    if symbolic_equal(o1_expr, correct_expr):
        o1_correct += 1
        data[index]["o1_correct"] = True
    else:
        data[index]["o1_correct"] = False


# Calculate accuracies
deepseek_accuracy = 100.0 * deepseek_correct / total
o1_accuracy = 100.0 * o1_correct / total

print(f"Deepseek Accuracy: {deepseek_accuracy:.2f}%")
print(f"O1 Accuracy:       {o1_accuracy:.2f}%")

###############################################################################
# Plotting with Seaborn
###############################################################################
sns.set_context("talk")  # Larger text
sns.set_style("whitegrid")  # White grid background
plt.figure(figsize=(8, 6))  # Increase figure size

# Create a small DataFrame for plotting
df = pd.DataFrame(
    {
        "Model": ["Deepseek R1", "OpenAI O1"],
        "Accuracy": [deepseek_accuracy, o1_accuracy],
    }
)

# Draw the barplot
bar_plot = sns.barplot(data=df, x="Model", y="Accuracy", palette="viridis")

# Annotate each bar with exact accuracy
for patch in bar_plot.patches:
    height = patch.get_height()
    bar_plot.annotate(
        f"{height:.1f}%",
        (patch.get_x() + patch.get_width() / 2, height),
        ha="center",
        va="bottom",
        xytext=(0, 5),  # Offset text slightly
        textcoords="offset points",
    )

bar_plot.set_ylim(0, 100)
bar_plot.set_ylabel("Accuracy (%)")
plt.subplots_adjust(top=0.90)  # Add more space at the top
bar_plot.set_title(
    "Deepseek R1 vs. OpenAI O1 on High School Calculus",
    fontsize=16,
    weight="bold",
    pad=50,
)

plt.show()
plt.savefig("model_comparison_plot.png", dpi=300)

##############################################################################
# Convert results to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("model_comparison_results.csv", index=False)
