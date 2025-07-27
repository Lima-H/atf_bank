import streamlit as st
from pdf2image import convert_from_bytes
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
import tempfile
from modelo import analisar_extrato_por_links

def format_currency(value):
    """Formata valores para moeda brasileira"""
    if isinstance(value, (int, float)):
        return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"R$ {str(value)}"

# Configurações googledrive
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = "1kvWh4CxWZsmovOBZat7QzgY9kw26o7RE"  # Minha pasta

def authenticate():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def upload_image_and_get_public_link(path_image, folder_id, return_id=False):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': os.path.basename(path_image),
        'parents': [folder_id]
    }
    media = MediaFileUpload(path_image, mimetype='image/png')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    # Torna o arquivo público
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(fileId=file_id, body=permission).execute()

    # Gera o link público
    public_url = f"https://drive.google.com/uc?id={file_id}"
    if return_id:
        return public_url, file_id
    return public_url

# --- INTERFACE PRINCIPAL ---
st.set_page_config(page_title="PDF para Imagens no Drive", page_icon="📄", layout="wide")
st.title("📄 Análise de Extratos Bancários")

# --- SIDEBAR ---
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
    1. Faça upload do arquivo PDF
    2. Clique em "Processar Extrato com IA"
    3. Aguarde o processamento
    4. Veja o resumo e as tabelas detalhadas
    """)


st.markdown("Faça upload de um PDF e processe automaticamente o extrato bancário.")

uploaded_file = st.file_uploader("Selecione o PDF", type=['pdf'])

if uploaded_file is not None:
    st.success(f"Arquivo carregado: {uploaded_file.name}")

    if st.button("🚀 Processar Extrato com IA"):
        with st.spinner("🔄 Convertendo PDF, enviando imagens e analisando extrato... Isso pode levar alguns minutos."):
            # 1. Converter PDF em imagens
            imagens = convert_from_bytes(uploaded_file.read())
            links_publicos = []
            file_ids = []
            for i, img in enumerate(imagens):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img.save(tmp.name, format="PNG")
                    tmp.flush()
                    link, file_id = upload_image_and_get_public_link(tmp.name, FOLDER_ID, return_id=True)
                    links_publicos.append(link)
                    file_ids.append(file_id)
                os.unlink(tmp.name)
            # 2. Dados do modelo
            total_credito, total_debito, total_liquido, df, df_credito, df_debito, soma_valores_credito, soma_valores_debito, total_tokens, preco_total = analisar_extrato_por_links(links_publicos)
        
        st.success("✅ Processamento concluído!")

        # --- ORIGEM MAIS FREQUENTE ---
        origem_mais_frequente_credito = df_credito['origem'].mode()[0] if not df_credito.empty else ""
        origem_mais_frequente_debito = df_debito['origem'].mode()[0] if not df_debito.empty else ""

        # --- CARDS DE RESUMO ---
        st.subheader("📊 Resumo Financeiro")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="💰 Total Créditos",
                value=format_currency(total_credito),
                delta=f"{(df['tipo'] == 'credito').sum()} transações"
            )
        with col2:
            st.metric(
                label="💸 Total Débitos",
                value=format_currency(total_debito),
                delta=f"{(df['tipo'] == 'debito').sum()} transações"
            )
        with col3:
            st.metric(
                label="⚖️ Saldo Líquido",
                value=format_currency(total_liquido),
                delta=format_currency(abs(total_liquido)) if total_liquido != 0 else "Equilibrado"
            )

        # --- CARDS DE ORIGEM MAIS FREQUENTE ---
        st.markdown("#### Origem mais frequente")
        col4, col5 = st.columns(2)
        with col4:
            st.metric(
                label=f"🔝 Crédito: {origem_mais_frequente_credito}",
                value=format_currency(soma_valores_credito)
            )
        with col5:
            st.metric(
                label=f"🔝 Débito: {origem_mais_frequente_debito}",
                value=format_currency(soma_valores_debito)
            )

        # --- TABELA DETALHADA ---
        st.markdown("---")
        st.subheader("📄 Movimentações Detalhadas")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Créditos")
            st.dataframe(df_credito, use_container_width=True)
        with col2:
            st.markdown("### Débitos")
            st.dataframe(df_debito, use_container_width=True)
        
        # # --- INFORMAÇÕES DO PROCESSAMENTO ---
        # st.markdown("---")
        # st.subheader("💡 Informações do Processamento")

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.metric("🔢 Tokens Utilizados", f"{total_tokens:,.0f}")
        # with col2:
        #     st.metric("💵 Custo Estimado", f"${preco_total:.4f}")

        # --- DELETA AS IMAGENS DO GOOGLE DRIVE ---
        creds = authenticate()
        service = build('drive', 'v3', credentials=creds)
        for file_id in file_ids:
            try:
                service.files().delete(fileId=file_id).execute()
            except Exception as e:
                st.warning(f"Não foi possível deletar o arquivo {file_id}: {e}")