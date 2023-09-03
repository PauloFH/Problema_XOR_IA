import numpy as np

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoide
def sigmoid_derivada(x):
    return x * (1 - x)

# Função degrau
def funcao_degrau(x):
    return 1 if x >= 0.5 else 0


# Configuração da rede neural
tamanho_entrada = 2
tamanho_oculta = 2
tamanho_saida = 1
taxa_aprendizado = 0.1
epocas = 10000

# Passo 0
np.random.seed(0)
pesos_entrada_oculta = np.random.uniform(size=(tamanho_entrada, tamanho_oculta))
pesos_oculta_saida = np.random.uniform(size=(tamanho_oculta, tamanho_saida))


# Passo 1
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas_esperadas = np.array([[0], [1], [1], [0]])

# Treinamento da rede neural
for epoca in range(epocas):
    erro_total = 0

    for i in range(len(entradas)):
        
        # Passo 2
        entrada_oculta = np.dot(entradas[i], pesos_entrada_oculta)
        
        # Passo 3
        saida_oculta = sigmoid(entrada_oculta)
        
        # Passo 3
        entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
        
        # Passo 4
        saida_prevista = sigmoid(entrada_saida)
    
        # Passo 5
        erro = saidas_esperadas[i] - saida_prevista
        erro_total += np.sum(erro ** 2)
    
    
        # Passo 6
        d_saida = erro * sigmoid_derivada(saida_prevista)
        
        # Passo 7
        d_oculta = d_saida.dot(pesos_oculta_saida.T) * sigmoid_derivada(saida_oculta)
    
        # Atualização dos pesos
        pesos_oculta_saida += np.outer(saida_oculta, d_saida) * taxa_aprendizado
        pesos_entrada_oculta += np.outer(entradas[i], d_oculta) * taxa_aprendizado
    if epoca % 1000 == 0:
        print(f"Época {epoca}, Erro Médio: {erro_total / len(entradas)}")
temp = True
array = [0, 0]

while(temp):
    t = int(input("Quer testar novamente? 1 - sim / 0 - não  :"))
    if(t == 1):
        for i in range(2):
            array[i] = int(input("Digite o Número do Xor (0 ou 1):  "))
        entrada_oculta = np.dot(array, pesos_entrada_oculta)
        saida_oculta = sigmoid(entrada_oculta)
        entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
        saida_prevista = sigmoid(entrada_saida)
        saida_binaria = funcao_degrau(saida_prevista[0])
        print(f"Entrada: {array}, Saída prevista: {saida_binaria}")   
        temp = True
    else: 
        temp = False