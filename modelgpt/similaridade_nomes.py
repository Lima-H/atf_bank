
from difflib import SequenceMatcher
from collections import Counter

def similaridade_nomes(nome1, nome2, threshold=0.8):
    """
    Calcula a similaridade entre dois nomes usando SequenceMatcher.
    Retorna True se a similaridade for maior que o threshold.
    """
    # Normaliza os nomes (minúsculas, remove espaços extras)
    nome1 = ' '.join(nome1.lower().split())
    nome2 = ' '.join(nome2.lower().split())
    
    # Calcula similaridade
    similaridade = SequenceMatcher(None, nome1, nome2).ratio()
    return similaridade >= threshold

def agrupar_nomes_similares(nomes, threshold=0.8):
    """
    Agrupa nomes similares e retorna um dicionário de mapeamento.
    A chave é o nome original, o valor é o nome mais frequente do grupo.
    """
    nomes_unicos = list(set(nomes))
    grupos = []
    processados = set()
    
    for nome in nomes_unicos:
        if nome in processados:
            continue
            
        grupo_atual = [nome]
        processados.add(nome)
        
        for outro_nome in nomes_unicos:
            if outro_nome not in processados and similaridade_nomes(nome, outro_nome, threshold):
                grupo_atual.append(outro_nome)
                processados.add(outro_nome)
        
        grupos.append(grupo_atual)
    
    # Para cada grupo, escolhe o nome mais frequente
    mapeamento = {}
    for grupo in grupos:
        # Conta frequência de cada nome no grupo dentro da lista original
        contador = Counter([nome for nome in nomes if nome in grupo])
        nome_mais_frequente = contador.most_common(1)[0][0]
        
        # Mapeia todos os nomes do grupo para o mais frequente
        for nome in grupo:
            mapeamento[nome] = nome_mais_frequente
    
    return mapeamento

def normalizar_origens(df, threshold=0.8):
    """
    Normaliza as origens no DataFrame aplicando similaridade de nomes.
    """
    df_normalizado = df.copy()
    
    # Pega todas as origens únicas
    origens = df_normalizado['origem'].dropna().tolist()
    
    # Cria mapeamento de nomes similares
    mapeamento = agrupar_nomes_similares(origens, threshold)
    
    # Aplica o mapeamento
    df_normalizado['origem'] = df_normalizado['origem'].map(mapeamento).fillna(df_normalizado['origem'])
    
    return df_normalizado