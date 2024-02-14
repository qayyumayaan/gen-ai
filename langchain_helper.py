from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import random

load_dotenv()

def generate_advice(situation):
    llm = OpenAI(temperature=0.7)

    # Load the Art of War text
    with open("art_of_war.txt", "r", encoding="utf-8") as file:
        art_of_war_text = file.read()

    # Randomly select a passage from the text to base the advice on
    passages = art_of_war_text.split("\n\n")
    selected_passage = random.choice(passages)

    prompt_template_advice = PromptTemplate(
        input_variables=['situation', 'selected_passage'],
        template="Given the situation: '{situation}', what advice from this passage of Sun Tzu's Art of War would be most applicable? Passage: '{selected_passage}'. Advice: "
    )
    advice_chain = LLMChain(llm=llm, prompt=prompt_template_advice, output_key="advice")

    response = advice_chain({'situation': situation, 'selected_passage': selected_passage})
    
    # Include the selected passage in the response for citation
    response['selected_passage'] = selected_passage
    return response

if __name__ == "__main__":
    situation = "Negotiating a business deal"
    print(generate_advice(situation))
