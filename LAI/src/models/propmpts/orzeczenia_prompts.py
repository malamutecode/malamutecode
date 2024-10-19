"""Module with prompts used for app."""

from data.text_preprocessing import Languages
from models.propmpts.prompt_register import register_prompt


@register_prompt(model_name='google/gemma-2b-it', language=Languages.english)
def prompt_v0_1():
    prompt = """Based on the following context items, please answer the query.
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible.
        Use the following examples as reference for the ideal answer style.
        \nExample 1:
        Query: What are the fat-soluble vitamins?
        Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are
        absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use.
        Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in
        calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage.
        Vitamin K is essential for blood clotting and bone metabolism.
        \nExample 2:
        Query: What are the causes of type 2 diabetes?
        Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories
        leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin
        resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas
        cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally,
        excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain
        and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
        \nExample 3:
        Query: What is the importance of hydration for physical performance?
        Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume,
        regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration
        is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased
        performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking
        sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
        \nNow use the following context items to answer the user query:
        {context}
        \nRelevant passages: <extract relevant passages from the context here>
        User query: {query}
        Answer:"""
    return prompt


@register_prompt(model_name='speakleash/Bielik-11B-v2.3-Instruct-GPTQ', language=Languages.polish)
@register_prompt(model_name='speakleash/Bielik-11B-v2.3-Instruct', language=Languages.polish)
def prompt_v1():
    prompt = """
            Odpowiedz na pytanie bazując na podanym poniżej kontekście. Odpowiadaj w tonie formalnym. Działaj jako
            ekspert prawa. Celem jest wyciągnięcie z tekstu najważniejszych dla prawnika informacji.  Staraj się
            odpowiadać na podstawie kontekstu. Jeśli kontekst nie dotyczy pytania użytkownika - poinformuj o tym.
            Wyodrębniając odpowiednie fragmenty z kontekstu przed udzieleniem odpowiedzi na zapytanie. Nie zwracaj
            myślenia, tylko odpowiedź. Użyj poniższych przykładów jako odniesienia dla idealnego stylu odpowiedzi.
            \nPrzykład 1
            Pytanie użytkownika: Kradzież w sklepie - umorzenie sprawy.
            Odpowiedź: W sprawie jednej ze spraw, sąd rejonowy umorzył postępowanie dotyczące kradzieży sklepowej,
            uznając, że wartość skradzionego mienia (300 zł) była znikoma, a społeczna szkodliwość czynu była niewielka.
            Decyzja była oparta na art. 17 § 1 pkt 3 Kodeksu postępowania karnego, który mówi, że postępowanie nie
            może być prowadzone, jeśli społeczna szkodliwość czynu jest znikoma. W innym przypadku, sąd umorzył
            postępowanie z uwagi na zawarcie ugody między stronami, gdzie oskarżony naprawił szkodę, a pokrzywdzony
            wnioskował o umorzenie sprawy.

            \nPrzykład 2
            Opcja testowa 1

            Pytanie: Nielegalne wydobywanie kruszcu.
            Odpowiedź: W sprawie jednej ze spraw, sąd okręgowy skazał oskarżonego za nielegalne wydobywanie kruszców,
            zgodnie z art. 181 § 1 Kodeksu karnego, który penalizuje naruszenie zasad ochrony środowiska w zakresie
            wydobywania surowców naturalnych bez odpowiednich zezwoleń. Oskarżony został skazany na 1 rok pozbawienia
            wolności w zawieszeniu na 3 lata oraz nałożono na niego obowiązek zapłaty grzywny w wysokości 10 000 zł.
            W innym orzeczeniu, sąd orzekł karę 8 miesięcy pozbawienia wolności w zawieszeniu na 2 lata oraz obowiązek
            naprawienia szkody środowiskowej poprzez przywrócenie terenu do stanu sprzed wydobycia, po stwierdzeniu
            nielegalnego wydobywania piasku.

            Kontekst RAG
            \nTeraz użyj następującego kontekstu, aby odpowiedzieć na pytanie użytkownika:
            {context}
            \nPytanie użytkownika: {query}
            """
    return prompt
