from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

# Models
en_es_model_name = "Helsinki-NLP/opus-mt-en-es"
es_en_model_name = "Helsinki-NLP/opus-mt-es-en"

en_es_tokenizer = MarianTokenizer.from_pretrained(en_es_model_name)
en_es_model = MarianMTModel.from_pretrained(en_es_model_name)

es_en_tokenizer = MarianTokenizer.from_pretrained(es_en_model_name)
es_en_model = MarianMTModel.from_pretrained(es_en_model_name)

def translate(text, direction):
    if direction == "English to Spanish":
        tokenizer = en_es_tokenizer
        model = en_es_model
    else:
        tokenizer = es_en_tokenizer
        model = es_en_model

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter Text"),
        gr.Radio(["English to Spanish", "Spanish to English"])
    ],
    outputs="text",
    title="Language Translator"
)

iface.launch()
