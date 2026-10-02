import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:4b"


def convert_currency(amount, from_currency, to_currency):
    # Hardcoded rates for demo (ECB reference rates, 2 Oct 2026); in production you'd call a real exchange rate API
    rates = {
        "EUR-USD": 1.1225,
        "USD-EUR": 0.89087,
        "GBP-USD": 1.32008,
        "USD-GBP": 0.75753,
        "JPY-USD": 0.00634236,
        "USD-JPY": 157.67,
        "MXN-USD": 0.0545405,
        "USD-MXN": 18.335,
        "TRY-USD": 0.0203479,
        "USD-TRY": 49.145
    }

    rate_key = f"{from_currency}-{to_currency}"
    rate = rates.get(rate_key, 1.0)
    result = amount * rate

    return {
        "converted_amount": round(result, 2),
        "rate": rate,
        "result_text": f"{amount} {from_currency} = {result:.2f} {to_currency}"
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another at today's exchange rate. Use it for any question about exchange rates, including whether an exchange someone was offered was a good deal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount to convert"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "The source currency code (e.g., EUR, USD, GBP, JPY, MXN, TRY)"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "The target currency code (e.g., EUR, USD, GBP, JPY, MXN, TRY)"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]


def chat_with_tools(user_message):
    print(f"USER: {user_message}\n")

    messages = [{"role": "user", "content": user_message}]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": False,
        }
    )

    assistant_message = response.json().get("message", {})
    tool_calls = assistant_message.get("tool_calls", [])

    if tool_calls:
        print(f"Calling {len(tool_calls)} function(s)\n")

        messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]

            print(f"Function: {function_name}")
            print(f"Arguments: {json.dumps(function_args)}")

            if function_name == "convert_currency":
                result = convert_currency(
                    amount=function_args["amount"],
                    from_currency=function_args["from_currency"],
                    to_currency=function_args["to_currency"]
                )

                print(f"Result: {result['result_text']}\n")

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })

        final_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "think": False,
            }
        )

        final_message = final_response.json().get("message", {}).get("content", "")
        print(f"ASSISTANT: {final_message}\n")

    else:
        content = assistant_message.get("content", "")
        print(f"ASSISTANT: {content}\n")


if __name__ == "__main__":
    print(f"Model: {MODEL}\n")
    print("-" * 60)

    chat_with_tools("What is EUR 50 in USD?")
    print("-" * 60)

    chat_with_tools("Convert 100 USD to GBP")
    print("-" * 60)

    chat_with_tools("How many US dollars is 500 Mexican pesos?")
    print("-" * 60)

    chat_with_tools("How many Turkish lira do I get for 200 dollars?")
    print("-" * 60)

    # Needs the tool AND some reasoning: £100 is worth about $132, so $120 was a poor rate. Run it a few times.
    # Then try "I was given $120 for £100, was this good?": small models often convert the wrong way round.
    chat_with_tools("I changed £100 into dollars and got $120. Was that a good rate?")
    print("-" * 60)

    chat_with_tools("What is the capital of France?")
    print("-" * 60)
