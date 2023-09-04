import numpy as np
import sigmoid as s


# Configuração da rede neuralusado em coleções para combinar seus elementos em um único valor, aplicando uma função cumulativa que você especifica  
tamanho_entrada = 2
tamanho_oculta = 4
tamanho_saida = 1
taxa_aprendizado = 0.35
epocas = 10000
np.random.seed(0)


# Inicializa os pesos com valores aleatórios
pesos_entrada_oculta = np.random.uniform(size=(tamanho_entrada, tamanho_oculta))
pesos_oculta_saida = np.random.uniform(size=(tamanho_oculta, tamanho_saida))

# Aplica o valor das entradas e saídas esperadas
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas_esperadas = np.array([[0], [1], [1], [0]])


# Treinamento da rede neural
for epoca in range(epocas):
    erro_total = 0

    for i in range(len(entradas)):
        
        # Calculam-se os nets dos neurônios da camada oculta
        entrada_oculta = np.dot(entradas[i], pesos_entrada_oculta)
        
        # Aplica a função de ativação sigmoide para obter a saida da camada oculta
        saida_oculta = s.sigmoid(entrada_oculta)
        
        # Calcula os os nets dos neurônios da camada de saída
        entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
        
        # Calcula as saidas dos neuronios da camada de saída
        saida_prevista = s.sigmoid(entrada_saida)

        #  Calcula o erro da camada de saída
        erro = saidas_esperadas[i] - saida_prevista
        # INÍCIO DO BACKPROPAGATION
        d_saida = erro * s.sigmoid_derivada(saida_prevista)
        
        # Calcula-se os erros nos neurônios da camada oculta
        d_oculta = d_saida.dot(pesos_oculta_saida.T) * s.sigmoid_derivada(saida_oculta)
    
        # Atualição dos pesos da camada de saída
        pesos_oculta_saida += np.outer(saida_oculta, d_saida) * taxa_aprendizado
        
        # Atualiza os pesos da camada oculta
        pesos_entrada_oculta += np.outer(entradas[i], d_oculta) * taxa_aprendizado
        
        # Calcula o erro total da rede
        erro_total += np.sum(erro ** 2)
        
    if epoca % 1000 == 0:
        print(f"Época {epoca}, Erro Médio: {erro_total / len(entradas)}")
        

# Função que irá transformar a saida em 0 ou 1
def funcao_binaria(x):
    return 1 if x >= 0.5 else 0

# Teste da rede neural

array = [0, 0]
entrada_oculta = np.dot(array, pesos_entrada_oculta)
saida_oculta = s.sigmoid(entrada_oculta)
entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
saida_prevista = s.sigmoid(entrada_saida)
saida_binaria = funcao_binaria(saida_prevista[0])
print(f"Entrada: {array}, Saída prevista: {saida_prevista}")   
array = [1, 0]
entrada_oculta = np.dot(array, pesos_entrada_oculta)
saida_oculta = s.sigmoid(entrada_oculta)
entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
saida_prevista = s.sigmoid(entrada_saida)
saida_binaria = funcao_binaria(saida_prevista[0])
print(f"Entrada: {array}, Saída prevista: {saida_prevista}")   
array = [0, 1]
entrada_oculta = np.dot(array, pesos_entrada_oculta)
saida_oculta = s.sigmoid(entrada_oculta)
entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
saida_prevista = s.sigmoid(entrada_saida)
saida_binaria = funcao_binaria(saida_prevista[0])
print(f"Entrada: {array}, Saída prevista: {saida_prevista}")   
array = [1, 1]
entrada_oculta = np.dot(array, pesos_entrada_oculta)
saida_oculta = s.sigmoid(entrada_oculta)
entrada_saida = np.dot(saida_oculta, pesos_oculta_saida)
saida_prevista = s.sigmoid(entrada_saida)
saida_binaria = funcao_binaria(saida_prevista[0])
print(f"Entrada: {array}, Saída prevista: {saida_prevista}")   

