from answer_generator import generate_answer

query = "What are the benefits of PM Kisan?"

retrieved_chunks = [
    """
    PM Kisan provides financial assistance of ₹6000 per year
    to eligible farmers.

    The amount is transferred directly into bank accounts.

    Small and marginal farmers are eligible.
    """
]

answer = generate_answer(query, retrieved_chunks)

print("\nGenerated Answer:")
print(answer)