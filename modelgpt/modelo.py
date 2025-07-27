import os
from dotenv import load_dotenv
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import io
from similaridade_nomes import normalizar_origens

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4.1-mini-2025-04-14",  # gpt-4.1-mini-2025-04-14     "o4-mini-2025-04-16"
    # temperature=0,
)

prompt = """
Você receberá imagens de um extrato de uma conta bancária, onde os débitos podem ser representados pela cor vermelha, por um sinal de menos, pela letra 'D', ou por algum outro sinal, enquanto que os créditos podem ser representados pela cor azul, por um sinal de +, pela letra 'C', ou por algum outro sinal.
Geralmente os dados estão estruturados, e seguem a forma de uma tabela. Entenda qual coluna tem as informações desajadas. Por exemplo: Pode receber uma imagem que tem uma coluna com nome valor e outra com nome saldo, você não deve considerar os valores de saldo nas suas operações. 
Os valores dos débitos e outra com os valores dos créditos das movimentações financeiras desta conta. Desconsidere os dados de saldo bancário, considerando apenas as movimentações.
1. O formato deve ser csv do tipo:    

tipo,valor,origem, data
debito,22.97,PIX QRS IFOOD.COM A27/03,28/03/2025
debito,25.11,PIX QRS SHPP BRASIL26/03,04/05/2025
credito,12.19,DEV PIX SHPP BRASIL28/03,05/05/2025

2.Se possível, na coluna origem, coloque apenas o nome da pessoa ou empresa para quem se refere a movimentação. Para exemplificar, o csv acima no melhor formato seria:
tipo,valor,origem, data
debito,22.97,IFOOD.COM,28/03/2025
debito,25.11,SHPP BRASIL,04/05/2025
credito,12.19,SHPP BRASIL,05/05/2025

Se não tiver nenhum nome de referência, apenas escreva como está no extrato

Caso não encontre algum campo deixe vazio;
*IMPORTANTE*: Considere apenas as movimentações financeiras. Não inclua dados de saldo bancário. 
Em alguns tipo de extrato, pode ser que tenha o "SALDO DO DIA" na coluna de movimentações, você NÃO DEVE colocar essa linha no csv.
"""

def analisar_extrato_por_links(links, threshold_similaridade = 0.8):
    """
    Recebe uma lista de links públicos de imagens (Google Drive),
    retorna total_credito, total_debito, total_liquido e o DataFrame.
    """
    csvs = []
    for link in links:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": link}}]
            )
        ]
        response = llm.invoke(messages)
        extrato_csv = response.content
        # Limpa possíveis mensagens extras do modelo, pega só o CSV
        match = re.search(r"tipo,valor,origem, data[\s\S]+", extrato_csv)
        if match:
            extrato_csv = match.group(0)
        csvs.append(extrato_csv)
    
    # Junta todos os CSVs em um só
    csv_final = "\n".join(csvs)
    df = pd.read_csv(io.StringIO(csv_final))
    # Garante que a coluna valor é float
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce').fillna(0)
    
    # NOVA FUNCIONALIDADE: Normaliza as origens usando similaridade
    df = normalizar_origens(df, threshold_similaridade)
    
    total_credito = df[df['tipo'] == 'credito']['valor'].sum()
    total_debito = df[df['tipo'] == 'debito']['valor'].sum()
    total_liquido = total_credito - total_debito
    df_credito =  df[df['tipo'] == 'credito'][['origem','valor','data']]
    df_debito =  df[df['tipo'] == 'debito'][['origem','valor','data']]
    df_credito.reset_index(inplace=True, drop=True)
    df_debito.reset_index(inplace=True, drop=True)
    soma_valores_credito = df_credito[df_credito['origem'] == df_credito['origem'].mode()[0]]['valor'].sum()
    soma_valores_debito = df_debito[df_debito['origem'] == df_debito['origem'].mode()[0]]['valor'].sum()

    # input_tokens = response.usage_metadata['input_tokens']  
    # output_tokens = response.usage_metadata['output_tokens']
    input_tokens = 0
    output_tokens = 0 
    total_tokens = input_tokens + output_tokens
    preco_total = input_tokens*0.40/1000000 + output_tokens*1.60/1000000

    return total_credito, total_debito, total_liquido, df, df_credito, df_debito, soma_valores_credito, soma_valores_debito, total_tokens, preco_total





