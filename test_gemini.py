from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6LRDRfmZwBSCze7DfLm5HDjXGA9mPb8rYJyiOKKfknVgA"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is PM-KISAN?"
)

print(response.text)