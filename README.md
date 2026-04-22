# Processo Seletivo – Intensivo Maker | AI


## **Relatório final do desafio**

👤 Identificação: **Dorian Dayvid Gomes Feitosa**


### 1️⃣ Resumo da Arquitetura do Modelo

O modelo de CNN modelado para esse projeto teve como foco principal a otimização de memória, mas ainda chegando a uma precisão geral de aproximadamente 98% antes da otimização, com perdas de menos de 0,5% após otimização, confiável para a maioria dos usos práticos em uma indústria. Como o dataset é extenso e muito bem estruturado, foi possível obter bons resultados sem requerer muita complexidade por parte do modelo. O modelo possui 3 camadas convolucionais, com uma camada de MaxPooling entre a primeira e a segunda, reduzindo a dimensionalidade do mapa de características. Por conta da simplicidade do objetivo do modelo e do foco em otimização, foi decidido não utilizar a técnica de dropout, observando que o modelo ainda possui bom desempenho, mesmo sem regularização. A camada densa responsável por tomar as decisões possui 32 neurônios. Apesar de ser menos do que normalmente é usado, essa decisão não comprometeu a precisão do modelo, e ainda resultou em uma diminuição considerável no tamanho antes da otimização. por fim, a IA Completou seu objetivo e terminou com um tamanho de 1040 KB antes da otimização, chegando a aproximadamente 90 KB depois de quantizada.



### 2️⃣ Bibliotecas Utilizadas

A lista integral das bibliotecas e dependências externas estão no `requirements.txt`, com destaque para as seguintes bibliotecas:

- tensorflow == 2.21.0
- numpy == 2.2.6
- keras == 3.12.1
- scikit-learn == 1.7.2

Obs: Algumas bibliotecas tiveram que regredir para versões anteriores para serem aprovadas no workflow.


### 3️⃣ Técnica de Otimização do Modelo

Após a avaliação de duas técnicas de otimização (Float32 e Dynamic Range), a técnica escolhida foi a de Dynamic Range, por apresentar uma redução considerável sem muitas perdas operacionais. Tal decisão partiu do seguinte resultado preliminar de tentativa de conversão usando as duas técnicas:

Tamanhos aproximados:
 - model.h5:              1.04 MB (HDF5)
 - teste_float32.tflite:  0.34 MB (TFLite float32)
 - teste_dynamic.tflite:  0.09 MB (TFLite dynamic)

Percebe-se claramente, portanto, os ganhos em memória do Dynamic Range.



### 4️⃣ Resultados Obtidos

O modelo treinado é eficiente e ocupa pouquíssimo espaço depois de otimizado com Dynamic Range, sendo adequado mesmo para ambientes com memória e processamento escassos. Possui limitações e cumpre estritamente o propósito proposto, mas excuta isso com qualidade. Suas características gerais são:

**ARQUITETURA DO MODELO**

Modelo: "sequential"

Sumário:

| Layer (type) | Output Shape | Param # |
| :--- | :--- | :--- |
| conv2d (Conv2D) | (None, 26, 26, 28) | 280 |
| max_pooling2d (MaxPooling2D) | (None, 13, 13, 28) | 0 |
| conv2d_1 (Conv2D) | (None, 11, 11, 28) | 7,084 |
| conv2d_2 (Conv2D) | (None, 9, 9, 28) | 7,084 |
| flatten (Flatten) | (None, 2268) | 0 |
| dense (Dense) | (None, 32) | 72,608 |
| dense_1 (Dense) | (None, 10) | 330 |

- Total de parâmetros: 87,388 (341.36 KB)
- Parâmetros treináveis: 87,386 (341.35 KB)
- Parâmetros não treináveis: 0 (0.00 B)
- Parâmetros do Optimizer: 2 (12.00 B)

**CONFIGURAÇÃO DE COMPILAÇÃO**

- Optimizer: Adam (com learning_rate=0.001)
- Loss: 'sparse_categorical_crossentropy'
- Metrics: ['accuracy']

**PARÂMETROS DO MODELO**

- Total de parâmetros: 87,386
- Total de camadas: 7
- Total de camadas convolucionais: 3
- Tamanho aproximado do modelo .h5:     1.04 MB (HDF5)
- Tamanho aproximado do modelo .tflite: 0.09 MB (TFLite dynamic)

**MÉTRICAS GERAIS (DO MODELO JÁ OTIMIZADO)**

- Precisão (geral): 0.9851
- Recall (geral): 0.9848
- Specificidade (geral): 0.9983
- F1-score (geral): 0.9849
- Latência média (100 execuções): 0.090 ms

Obs: Resultados podem variar para cada máquina.


### 5️⃣ Comentários Adicionais


- Decisões sobre tamanho e precisão do modelo foram desafiadoras, devido à falta de referência sobre a qualidade dos parâmetros. Porém, a precisão acima de 98% foi considerada aceitável para o modelo e baseado nisso foi usada a menor versão sem alterar demais os padrões.

- O modelo é eficiente para detecção de algarismos em escala de cinza, sendo preciso um tratamento para conversão de escala em cinza no caso de manuscritos coloridos, e pode ter problemas de classificação nas imagens de algarismos com muito ruído visual de fundo.