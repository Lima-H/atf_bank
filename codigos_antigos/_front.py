import streamlit as st
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
import tempfile
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv

# Configuração da página
st.set_page_config(
    page_title="Análise de Extratos Bancários",
    page_icon="🏦",
    layout="wide"
)

# Load environment variables
load_dotenv()

# === CONFIGURAÇÃO AWS ===
@st.cache_resource
def init_aws_client():
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
    
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION]):
        st.error("⚠️ Credenciais AWS não configuradas. Verifique o arquivo .env")
        st.stop()
    
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    os.environ['AWS_DEFAULT_REGION'] = AWS_DEFAULT_REGION
    
    return boto3.client('bedrock-runtime', region_name=AWS_DEFAULT_REGION)

MODEL_ID = 'arn:aws:bedrock:us-west-2:941100472524:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'

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

def analyze_image_raw(image_base64, prompt, bedrock_client):
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

        response = bedrock_client.invoke_model(
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
        st.error(f"Erro ao analisar imagem: {str(e)}")
        return '{}', 0, 0

def split_pdf(pdf_bytes, output_dir):
    try:
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_pdf_path = tmp_file.name

        doc = fitz.open(tmp_pdf_path)
        pdf_name = f"extrato_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        image_dir = os.path.join(output_dir, pdf_name)
        os.makedirs(image_dir, exist_ok=True)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            matrix = fitz.Matrix(2.0, 2.0)  # Zoom
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(os.path.join(image_dir, f"page{page_num + 1}.png"))

        doc.close()
        os.unlink(tmp_pdf_path)  # Remove o arquivo temporário
        return image_dir

    except Exception as e:
        st.error(f"Erro ao processar PDF: {e}")
        return None

def process_pdf(pdf_bytes, bedrock_client):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Processa o PDF
        pasta_imagens = split_pdf(pdf_bytes, temp_dir)
        
        if not pasta_imagens:
            return None, None, 0
        
        creditos = []
        debitos = []
        tokens_utilizados = 0
        
        # Progress bar
        image_files = [f for f in sorted(os.listdir(pasta_imagens)) if f.lower().endswith('.png')]
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, imagem_nome in enumerate(image_files):
            caminho_imagem = os.path.join(pasta_imagens, imagem_nome)
            status_text.text(f"Analisando página {i+1} de {len(image_files)}...")
            
            image_base64, image_bytes_len = image_to_base64(caminho_imagem)
            resposta_bruta, prompt_tokens, resposta_tokens = analyze_image_raw(
                image_base64, PROMPT_SIMPLIFICADO, bedrock_client
            )

            tokens_utilizados += (image_bytes_len/4 + prompt_tokens + resposta_tokens)

            try:
                json_str = re.search(r'\{.*\}', resposta_bruta, re.DOTALL)
                json_data = json.loads(json_str.group(0) if json_str else resposta_bruta.strip())

                debitos.extend(json_data.get('debito', []))
                creditos.extend(json_data.get('credito', []))
            except Exception as e:
                st.warning(f"Erro ao processar página {i+1}: {e}")
            
            progress_bar.progress((i + 1) / len(image_files))
        
        progress_bar.empty()
        status_text.empty()
        
        return debitos, creditos, tokens_utilizados

def format_currency(value):
    """Formata valores para moeda brasileira"""
    if isinstance(value, (int, float)):
        return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"R$ {str(value)}"

def create_summary_cards(debitos, creditos):
    """Cria cards de resumo"""
    total_debitos = sum(float(d) for d in debitos if str(d).replace('.', '', 1).replace('-', '', 1).isdigit())
    total_creditos = sum(float(c) for c in creditos if str(c).replace('.', '', 1).replace('-', '', 1).isdigit())
    saldo = total_creditos - total_debitos
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💸 Total Débitos",
            value=format_currency(total_debitos),
            delta=f"{len(debitos)} transações"
        )
    
    with col2:
        st.metric(
            label="💰 Total Créditos", 
            value=format_currency(total_creditos),
            delta=f"{len(creditos)} transações"
        )
    
    with col3:
        delta_color = "normal" if saldo >= 0 else "inverse"
        st.metric(
            label="⚖️ Saldo Líquido",
            value=format_currency(saldo),
            delta=format_currency(abs(saldo)) if saldo != 0 else "Equilibrado"
        )

# === INTERFACE STREAMLIT ===
def main():
    st.title("🏦 Análise de Extratos Bancários")
    st.markdown("---")
    
    # Inicializar cliente AWS
    bedrock_client = init_aws_client()
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.info("""
        Esta ferramenta analisa extratos bancários em PDF e extrai:
        - 💳 Valores de débitos
        - 💰 Valores de créditos
        - 📊 Resumo financeiro
        """)
        
        st.header("📋 Como usar")
        st.markdown("""
        1. Faça upload do arquivo PDF do extrato
        2. Aguarde o processamento
        3. Visualize os resultados
        4. Baixe o arquivo JSON
        """)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "📎 Selecione o arquivo PDF do extrato bancário",
        type=['pdf'],
        help="Carregue um arquivo PDF contendo o extrato bancário"
    )
    
    if uploaded_file is not None:
        # Informações do arquivo
        st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
        st.info(f"📁 Tamanho: {file_size:.2f} MB")
        
        # Botão de processamento
        if st.button("🚀 Processar Extrato", type="primary"):
            with st.spinner("🔄 Processando extrato bancário..."):
                debitos, creditos, tokens_utilizados = process_pdf(uploaded_file.getvalue(), bedrock_client)
            
            if debitos is not None and creditos is not None:
                st.success("✅ Processamento concluído!")
                
                # Cards de resumo
                st.subheader("📊 Resumo Financeiro")
                create_summary_cards(debitos, creditos)
                
                st.markdown("---")
                
                # Resultados detalhados em colunas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💸 Débitos")
                    if debitos:
                        df_debitos = pd.DataFrame({
                            'Valor': [format_currency(float(d)) if str(d).replace('.', '', 1).replace('-', '', 1).isdigit() else str(d) for d in debitos]
                        })
                        st.dataframe(df_debitos, use_container_width=True)
                    else:
                        st.info("Nenhum débito encontrado")
                
                with col2:
                    st.subheader("💰 Créditos")
                    if creditos:
                        df_creditos = pd.DataFrame({
                            'Valor': [format_currency(float(c)) if str(c).replace('.', '', 1).replace('-', '', 1).isdigit() else str(c) for c in creditos]
                        })
                        st.dataframe(df_creditos, use_container_width=True)
                    else:
                        st.info("Nenhum crédito encontrado")
                
                # Informações de custo
                model_cost = 0.003  # per 1k tokens
                custo_total = (tokens_utilizados / 1000) * model_cost
                
                st.markdown("---")
                st.subheader("💡 Informações do Processamento")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔢 Tokens Utilizados", f"{tokens_utilizados:,.0f}")
                with col2:
                    st.metric("💵 Custo Estimado", f"${custo_total:.4f}")
                
                # Preparar dados para download
                resultados = {
                    "debito": debitos,
                    "credito": creditos,
                    "resumo": {
                        "total_debitos": sum(float(d) for d in debitos if str(d).replace('.', '', 1).replace('-', '', 1).isdigit()),
                        "total_creditos": sum(float(c) for c in creditos if str(c).replace('.', '', 1).replace('-', '', 1).isdigit()),
                        "quantidade_debitos": len(debitos),
                        "quantidade_creditos": len(creditos),
                        "tokens_utilizados": tokens_utilizados,
                        "custo_processamento": custo_total,
                        "data_processamento": datetime.now().isoformat()
                    }
                }
                
                # Botão de download
                json_string = json.dumps(resultados, indent=4, ensure_ascii=False)
                st.download_button(
                    label="📥 Baixar Resultados (JSON)",
                    data=json_string,
                    file_name=f"extrato_analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            else:
                st.error("❌ Erro no processamento do arquivo. Verifique se o PDF é válido.")

if __name__ == "__main__":
    main()