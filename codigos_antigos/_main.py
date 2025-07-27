import fitz  # PyMuPDF
import os
import shutil
import boto3
import base64
from PIL import Image
import io
import json
import tiktoken
import re

from dotenv import load_dotenv
import os

load_dotenv()

# === CONFIGURAÇÃO AWS ===
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['AWS_DEFAULT_REGION'] = AWS_DEFAULT_REGION

MODEL_ID = 'arn:aws:bedrock:us-west-2:941100472524:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'
bedrock = boto3.client('bedrock-runtime', region_name=AWS_DEFAULT_REGION)

# === PROMPT PARA O MODELO ===
PROMPT_SIMPLIFICADO = """
A imagem representa um extrato de uma conta bancária, onde os débitos podem ser representados pela cor vermelha, por um sinal de menos, pela letra 'D', ou por algum outro sinal, enquanto que os créditos podem ser representados pela cor vermelha, por um sinal de +, pela letra 'C', ou por algum outro sinal. 
Retorne um JSON com uma lista dos valores dos débitos e outra com os valores dos créditos das movimentações financeiras desta conta. Desconsidere os dados de saldo bancário, considerando apenas as movimentações.
Retorne apenas o JSON puro, sem explicações, no seguinte formato:
{
    "debito": [...],
    "credito": [...]
}
"""

# === FUNÇÕES DE UTILIDADE ===

def contar_tokens(texto, model_name="claude-instant-1"):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(texto))

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), len(buffer.getvalue())

def analyze_image_raw(image_base64, prompt):
    try:
        prompt_tokens = contar_tokens(prompt)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 700,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }

        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )

        result = json.loads(response['body'].read())
        resposta = result['content'][0]['text']
        response_tokens = contar_tokens(resposta)
        return resposta, prompt_tokens, response_tokens
    except Exception as e:
        print(f"Erro ao analisar imagem: {str(e)}")
        return '{}', 0, 0

def split_pdf(pdf_path, output_dir):
    try:
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        image_dir = os.path.join(output_dir, pdf_name)
        os.makedirs(image_dir, exist_ok=True)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            matrix = fitz.Matrix(2.0, 2.0)  # Zoom
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(os.path.join(image_dir, f"page{page_num + 1}.png"))

        doc.close()
        shutil.copy(pdf_path, os.path.join(image_dir, os.path.basename(pdf_path)))
        return image_dir

    except Exception as e:
        print(f"Erro ao dividir PDF '{pdf_path}': {e}")
        return None







# === EXECUÇÃO DO PROCESSO PARA UM PDF ESPECÍFICO ===

# Caminho para o PDF único a ser processado
pdf_path = 'pdfs_exemplos/extratoiti.pdf'
output_dir = 'imagens'
creditos = []
debitos = []
tokens_utilizados = 0
# Divide o PDF em imagens
pasta_imagens = split_pdf(pdf_path, output_dir)

if pasta_imagens:
    for imagem_nome in sorted(os.listdir(pasta_imagens)):
        if not imagem_nome.lower().endswith('.png'):
            continue

        caminho_imagem = os.path.join(pasta_imagens, imagem_nome)
        print(f"Analisando imagem: {caminho_imagem}")
        
        image_base64, image_bytes_len = image_to_base64(caminho_imagem)
        resposta_bruta, prompt_tokens, resposta_tokens = analyze_image_raw(image_base64, PROMPT_SIMPLIFICADO)

        tokens_utilizados = tokens_utilizados + (image_bytes_len/4 + prompt_tokens + resposta_tokens)

        try:
            json_str = re.search(r'\{.*\}', resposta_bruta, re.DOTALL)
            json_data = json.loads(json_str.group(0) if json_str else resposta_bruta.strip())

            debitos.extend(json_data.get('debito', []))
            creditos.extend(json_data.get('credito', []))
        except Exception as e:
            print(f"Erro ao processar resposta do modelo: {e}")
            print("Resposta bruta:", resposta_bruta)

# === RESULTADOS FINAIS ===
model_cost = 0.003 #per 1k tokens
print("\n===== RESULTADOS FINAIS =====")
print("DÉBITOS:", debitos)
print("CRÉDITOS:", creditos)
print("Total de tokens utilizados:", tokens_utilizados)
print("Custo do modelos:", (tokens_utilizados/1000) * model_cost)


# Salva os resultados em um arquivo JSON
resultados = {
    "debito": debitos,
    "credito": creditos
}
with open(pasta_imagens+'\\resultados.json', 'w') as f:
    json.dump(resultados, f, indent=4)