from extract_feats import extract_numbers
from mistralai import Mistral


def extract_numbers_LLM(text, api_key="", temperature=1.):
    few_shots = ["""
        For example, here is a description: "This graph comprises 10 nodes and 5 edges. The average degree is equal to 1 and there are 0 triangles in the graph. The global clustering coefficient and the graph's maximum k-core are 0 and 1 respectively. The graph consists of 5 communities."
        And here is the expected answer: "10.0 5.0 1.0 0.0 0.0 1.0 5.0"
    """,
    """
        As another example, here is a description of another graph: "In this graph, there are 28 nodes connected by 375 edges. On average, each node is connected to 26.785714285714285 other nodes. Within the graph, there are 3199 triangles, forming closed loops of nodes. The global clustering coefficient is 0.9921430786725938. Additionally, the graph has a maximum k-core of 25 and a number of communities equal to 1."
        The expected answer is: "28.0 375.0 26.785714285714285 3199.0 0.9921430786725938 25.0 1.0"
    """]

    prompt = f"""
        From a description of a graph, I would like you to extract some corresponding features. The description contains informations on the number of nodes and edges, the average degree, the number of triangles, the global clustering coefficient, the graph's maximum k-core and the number of communities.
        The answer should take the form of a list of floats, in the following order: number of nodes, number of edges, average degree, number of triangles, global clustering coefficient, maximum k-core, number of communities.
        You should only return this list of number, seperated by blanck spaces and no other additional information. \n
        {few_shots[0]}
        {few_shots[1]}

        Here are the description of several graph: {text}

        As a reminder, you should only return the list of numbers, in the correct order, seperated by blanck spaces and no other additional information.
    """

    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=temperature
            )

    try:
        float_list = list(map(float, chat_response.choices[0].message.content.split()))
    except ValueError:
        float_list = extract_numbers(text)

    return float_list