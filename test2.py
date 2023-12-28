# Langchain 패키지들
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import BedrockChat
import boto3
import json
from langchain.document_loaders import YoutubeLoader
import streamlit as st
import requests
import pandas as pd
import os
import io
import base64
from io import BytesIO
from PIL import Image

session = boto3.Session()

bedrock = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
)

#이미지 생성
bedrock_model_id = "stability.stable-diffusion-xl-v0" # stability.stable-diffusion-xl-v1로 바꿔보세요.
#바이트 이미지 형태로 변환
def get_response_image_from_payload(response): 

    payload = json.loads(response.get('body').read()) 
    images = payload.get('artifacts')
    image_data = base64.b64decode(images[0].get('base64'))

    return BytesIO(image_data) 

def get_image_response(prompt_content): 
    
    request_body = json.dumps({"text_prompts": 
                               [ {"text": prompt_content } ], 
                               "cfg_scale": 9, 
                               "steps": 80, }) 
    #바이트 형태
    response = bedrock.invoke_model(body=request_body, modelId=bedrock_model_id)
    
    output = get_response_image_from_payload(response) 
     
    return output

st.title("영상 썸네일 제작 👨‍🎨")

st.image('https://wikidocs.net/images/page/215361/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%ED%99%94%EA%B0%80.png', width=200)

#유튜브 URL 넣는 칸
youtube_video_url=st.text_area('썸네일로 만들 영상 URL을 입력하세요.') #URL 넣는 칸 할당
a=st.button('printing')
if a:
        #유튜브 스크립트 추출하기
    loader = YoutubeLoader.from_youtube_url(youtube_video_url)
    transcript = loader.load()
    script=transcript[0].page_content
# 언어모델 설정
    llm = BedrockChat(model_kwargs={"temperature": 0},
                        model_id="anthropic.claude-v2",
                        client=bedrock
                    )
# 프롬프트 설정
    prompt = PromptTemplate(
    template="""백틱으로 둘러싸인 전사본을 이용해 해당 유튜브 비디오를 요약해주세요. \
    ```{text}```
    """, input_variables=["text"]
    )
    combine_prompt = PromptTemplate(
    template="""Combine all the youtube video transcripts provided within backticks \
    ```{text}```
    Provide a concise summary between 5 to 10 sentences.
    """, input_variables=["text"]
    )
# LangChain을 활용하여 긴 글 요약하기
# 글 쪼개기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    texts = text_splitter.create_documents([script])

# 요약하기
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False,
                            map_prompt=prompt, combine_prompt=combine_prompt)
    summerize = chain.run(texts)


#요약한 내용 넣어서 이미지 생성
    input_text = summerize
    if input_text:
        try:
            image = get_image_response(input_text)
            st.image(image)
        except:
            st.error("요청 오류가 발생했습니다")
        else:
            st.warning("텍스트를 입력하세요")