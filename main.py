import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from google_trans_new import google_translator  
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
from PIL import Image
from streamlit_mic_recorder import mic_recorder,speech_to_text
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# import logging
# import logging.handlers
# import queue
# import threading
# import time
# import urllib.request
# import os
# from collections import deque
# from pathlib import Path
# from typing import List

# import av
# import numpy as np
# import pydub
# import streamlit as st
# from twilio.rest import Client

# from streamlit_webrtc import WebRtcMode, webrtc_streamer
#from streamlit_mic_recorder import mic_recorder,speech_to_text
# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize the translator
translator = Translator()

# Language dictionary
language_dict = {
    'bengali': 'bn',
    'english': 'en',
    'gujarati': 'gu',
    'hindi': 'hi',
    'malayalam': 'ml',
    'marathi': 'mr',
    'nepali': 'ne',
    'odia': 'or',
    'punjabi': 'pa',
    'tamil': 'ta',
    'telugu': 'te',
}

# language_dict = {
#     'afrikaans': 'af',
#     'albanian': 'sq',
#     'amharic': 'am',
#     'arabic': 'ar',
#     'armenian': 'hy',
#     'azerbaijani': 'az',
#     'basque': 'eu',
#     'belarusian': 'be',
#     'bengali': 'bn',
#     'bosnian': 'bs',
#     'bulgarian': 'bg',
#     'catalan': 'ca',
#     'cebuano': 'ceb',
#     'chichewa': 'ny',
#     'chinese (simplified)': 'zh',
#     'chinese (traditional)': 'zh-TW',
#     'corsican': 'co',
#     'croatian': 'hr',
#     'czech': 'cs',
#     'danish': 'da',
#     'dutch': 'nl',
#     'english': 'en',
#     'esperanto': 'eo',
#     'estonian': 'et',
#     'filipino': 'tl',
#     'finnish': 'fi',
#     'french': 'fr',
#     'frisian': 'fy',
#     'galician': 'gl',
#     'georgian': 'ka',
#     'german': 'de',
#     'greek': 'el',
#     'gujarati': 'gu',
#     'haitian creole': 'ht',
#     'hausa': 'ha',
#     'hawaiian': 'haw',
#     'hebrew': 'he',
#     'hindi': 'hi',
#     'hmong': 'hmn',
#     'hungarian': 'hu',
#     'icelandic': 'is',
#     'igbo': 'ig',
#     'indonesian': 'id',
#     'irish': 'ga',
#     'italian': 'it',
#     'japanese': 'ja',
#     'javanese': 'jv',
#     'kannada': 'kn',
#     'kazakh': 'kk',
#     'khmer': 'km',
#     'kinyarwanda': 'rw',
#     'korean': 'ko',
#     'kurdish': 'ku',
#     'kyrgyz': 'ky',
#     'lao': 'lo',
#     'latin': 'la',
#     'latvian': 'lv',
#     'lithuanian': 'lt',
#     'luxembourgish': 'lb',
#     'macedonian': 'mk',
#     'malagasy': 'mg',
#     'malay': 'ms',
#     'malayalam': 'ml',
#     'maltese': 'mt',
#     'maori': 'mi',
#     'marathi': 'mr',
#     'mongolian': 'mn',
#     'myanmar (burmese)': 'my',
#     'nepali': 'ne',
#     'norwegian': 'no',
#     'odia': 'or',
#     'pashto': 'ps',
#     'persian': 'fa',
#     'polish': 'pl',
#     'portuguese': 'pt',
#     'punjabi': 'pa',
#     'romanian': 'ro',
#     'russian': 'ru',
#     'samoan': 'sm',
#     'scots gaelic': 'gd',
#     'serbian': 'sr',
#     'sesotho': 'st',
#     'shona': 'sn',
#     'sindhi': 'sd',
#     'sinhala': 'si',
#     'slovak': 'sk',
#     'slovenian': 'sl',
#     'somali': 'so',
#     'spanish': 'es',
#     'sundanese': 'su',
#     'swahili': 'sw',
#     'swedish': 'sv',
#     'tajik': 'tg',
#     'tamil': 'ta',
#     'telugu': 'te',
#     'thai': 'th',
#     'turkish': 'tr',
#     'ukrainian': 'uk',
#     'urdu': 'ur',
#     'uyghur': 'ug',
#     'uzbek': 'uz',
#     'vietnamese': 'vi',
#     'welsh': 'cy',
#     'xhosa': 'xh',
#     'yiddish': 'yi',
#     'yoruba': 'yo',
#     'zulu': 'zu',
# }


#img=Image.open('finallogo.jpg')

# col1, col2 = st.columns([1,3])

# #with col1:
#         #st.image(img, width=220)

# with col2:
#         custom_theme = {
#             "theme": {
#                 "primaryColor": "#000000",
#                 "backgroundColor": "#89939E",
#                 "secondaryBackgroundColor": "#262730",
#                 "textColor": "#FFFFFF",
#                 "font": "Serif"
#             }
#         }

#         # Apply custom theme to Streamlit
        # st.markdown(
        #     f"""
        #     <style>
        #     :root {{
        #         --primary-color: {custom_theme["theme"]["primaryColor"]};
        #         --background-color: {custom_theme["theme"]["backgroundColor"]};
        #         --secondary-background-color: {custom_theme["theme"]["secondaryBackgroundColor"]};
        #         --text-color: {custom_theme["theme"]["textColor"]};
        #         --font: {custom_theme["theme"]["font"]};
        #     }}
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )


        
#         st.title("AMPYtranslate")

# st.header("")
# hide_st_style = """
#                 <style>
#                 #Mainmenu{visibility: hidden;}
#                 footer{visibility:hidden; }
#                 </style>
#                 """

# st.markdown(hide_st_style, unsafe_allow_html=True)




# @st.cache_data  # type: ignore
# def get_ice_servers():
#     """Use Twilio's TURN server because Streamlit Community Cloud has changed
#     its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
#     We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
#     but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
#     See https://github.com/whitphx/streamlit-webrtc/issues/1213
#     """

#     # Ref: https://www.twilio.com/docs/stun-turn/api
#     try:
#         account_sid = "ACbea2776671d07a28bfa473b522b609fb"
#         auth_token = "f9d226593c16124f8eaed4b8d2d5397b"
#     except KeyError:
#         logger.warning(
#             "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
#         )
#         return [{"urls": ["stun:stun.l.google.com:19302"]}]

#     client = Client(account_sid, auth_token)

#     token = client.tokens.create()

#     return token.ice_servers

# Function to recognize speech
# def recognize_speech(prompt, language='en'):
#     try:
#         # audio_source = mic_recorder(
#         #             start_prompt="Start recording",
#         #             stop_prompt="Stop recording", 
#         #             just_once=False,
#         #             use_container_width=False,
#         #             key='recorder')
#         with sr.Microphone() as source:
#             st.write(prompt)
#             # Use recognizer to adjust to ambient noise
#             recognizer.adjust_for_ambient_noise(source, duration=1)
#             audio = recognizer.listen(source)
#             st.write("Recognizing...")

#         # Recognize the speech in the specified language
#         spoken_text = recognizer.recognize_google(audio, language=language)
#         return spoken_text
#     except sr.UnknownValueError:
#         st.write("Sorry, I couldn't understand your speech.")

        

# Function to translate speech
def translate_speech():
    
    #source_language_name = recognize_speech("Please speak the source language name (e.g., 'English'): ")
    
        
    st.title("BITranSlate")
    # st.write("Record your voice, and play the recorded audio:")
    # audio=mic_recorder(start_prompt="⏺️",stop_prompt="⏹️",key='recorder')

    custom_theme = {
            "theme": {
                "primaryColor": "#000000",
                "backgroundColor": "#89939E",
                "secondaryBackgroundColor": "#262730",
                "textColor": "#FFFFFF",
                "font": "Serif"
            }
        }
    st.markdown(
            f"""
            <style>
            :root {{
                --primary-color: {custom_theme["theme"]["primaryColor"]};
                --background-color: {custom_theme["theme"]["backgroundColor"]};
                --secondary-background-color: {custom_theme["theme"]["secondaryBackgroundColor"]};
                --text-color: {custom_theme["theme"]["textColor"]};
                --font: {custom_theme["theme"]["font"]};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    source_language_name = st.selectbox('Please input the source language',language_dict)
    source_language = language_dict[source_language_name]
    target_language_name = st.selectbox('Please input the target language',language_dict)
    target_language = language_dict[target_language_name]

    c1,c2=st.columns(2)
    with c1:
        st.write("Convert speech to text:")
    with c2:
        text=speech_to_text(language=source_language,use_container_width=True,just_once=True,key='STT')

    sentence = text
    nllb_langs = {'hindi':'hin_Deva',
                  'english':'eng_Latn',
                  'punjabi':'pan_Guru',
                  'odia':'ory_Orya',
                  'bengali':'ben_Beng',
                  'telugu':'tel_Tulu',
                  'tamil':'tam_Taml',
                  'nepali':'npi_Deva',
                  'marathi':'mar_Deva',
                  'malayalam':'mal_Mlym',
                  'kannada':'kan_Knda',
                  'gujarati':'guj_Gujr',
                  }
    # translator_google = Translator(service_urls=[
    #   'translate.googleapis.com'
    # ])
    #translator_google = google_translator()
    translator = pipeline('translation', model=AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M"), tokenizer=AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"), src_lang=nllb_langs[source_language_name], tgt_lang=nllb_langs[target_language_name], max_length = 4000)
    text_to_translate = text
    translated_text = translator(text_to_translate)[0]['translation_text']
    translated_text_google = GoogleTranslator(source='auto', target=target_language).translate(text_to_translate)
    #translated_text_google = translator_google.translate(text_to_translate, lang_tgt=target_language)

    # translated_text_google = translator_google.translate(text_to_translate, src=source_language, dest=target_language)

    #translated_text_google = translator_google.translate(text_to_translate, src=source_language, dest=target_language)
    model2 = SentenceTransformer("google/muril-base-cased")
            # Compute embeddings for the sentences
    embedding = model2.encode(text_to_translate, convert_to_tensor=True)
    embeddings_nllb = model2.encode(translated_text, convert_to_tensor=True)
    embeddings_google = model2.encode(translated_text_google, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_score_nllb = util.cos_sim(embedding, embeddings_nllb).item()
    cosine_score_google = util.cos_sim(embedding, embeddings_google).item()

    # Select the translation with the higher cosine similarity score
    selected_translation = translated_text if cosine_score_nllb > cosine_score_google else translated_text_google

    st.write(f"Source Language: {source_language_name}")
    st.write(f"Sentence: {sentence}")
    st.write(f"Destination Language: {target_language_name}")
    st.write(f"Translated Text from NLLB: {translated_text}")
    st.write(f"Translated Text from Google Translate: {translated_text_google}")
    st.write(f"More accurate translation: {selected_translation}")

    # Using Google-Text-to-Speech to speak the translated text
    speak = gTTS(text=translated_text, lang=target_language, slow=False)
    #speak.save("translated_voice.mp3")

    # Play the translated voice
    #playsound('translated_voice.mp3')

#if st.button("  CLICK HERE TO TRANSLATE  "):
translate_speech()
