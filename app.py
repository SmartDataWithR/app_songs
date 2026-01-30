#%% pakete
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# %% Output Parser
class MusicBandAlbum(BaseModel):
    band_name: str
    song_name: str
    band_members: list[str]
    album_name: str
    sales: str = Field(description="sales count of physical devices", examples=["60 million copies"])
    release_year: int 

class MusicBandAlbumList(BaseModel):
    result: list[MusicBandAlbum]

parser = PydanticOutputParser(pydantic_object=MusicBandAlbum)

from pprint import pprint
pprint(parser.get_format_instructions())

#%% prompt template
messages = [
    ("system", """
        Du bist Musikexperte und lieferst Informationen zu bestimmten Songs.
        Der Nutzer übergibt dir eine Songzeile und du gibst in strukturierter Form Informationen zurück.
        Halte dich dabei an das vorgegebene Schema {schema}
    """),
    ("user", "Songzeile: {song_line}")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages).partial(schema=parser.get_format_instructions())
prompt_template

# %% Modell
MODEL_NAME = "gpt-4o"
model = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.0)


#%% Chain erstellen
chain = prompt_template | model | parser

def get_model_output(user_prompt, history):
    res = chain.invoke({"song_line": user_prompt})
    res = res.model_dump()
    print(f"Res: {res}")

    output = f"""
    **Band:** {res['band_name']} \n 
    **Song:** {res['song_name']} \n 
    **Mitglieder:** {', '.join(res['band_members'])} \n 
    **Album:** {res['album_name']} \n 
    **Verkäufe:** {res['sales']} \n 
    **Jahr:** {res['release_year']}
    """
    return output
    
if __name__ == "__main__":
    # Beispiel für Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("## Musikexperten-Chatbot: Nenne eine Songzeile!")
        chat = gr.ChatInterface(fn=get_model_output)

    demo.launch()