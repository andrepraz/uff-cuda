# Redes neurais para classificação de textos jornalísticos e processamento paralelo em CUDA e PyTorch

### André Ricardo Prazeres Rodrigues

## INTRODUÇÃO

A evolução das tecnologias de informação, a comunicação entre as pessoas, abriu novas oportunidades e plataformas de interação. O que possibilitou o surgimento de novos canais de mídia e produção de conteúdo (SILVA; TESSAROLO, 2016).
As redes sociais estão se tornando um padrão para a comunicação e a troca de informações. Mas suas publicações são limitadas no que se refere ao número de caracteres postados. A comunicação do texto deve ser mais concisa e isso independente do tema: entretenimento, esporte, negócio, política, tecnologia etc.
DUARTE; RIVOIRE; AUGUSTO RIBEIRO (2016) estudaram como as redes sociais são fornecedoras de pauta, geradoras de conteúdo e disseminadoras da produção jornalística.

[em construção...]

Neste contexto, o objetivo é classificar os textos jornalísticos com base nos temas entretenimento, esporte, negócio, política, tecnologia com redes neurais aplicando o processamento paralelo em CUDA com o módulo PyTorch.

## REDES NEURAIS
A linguagem natural é utilizada pelos humanos que é diferente de uma linguagem de programação. A linguagem natural trabalha com palavras e a linguagem de programação trabalha com números. Então, a tarefa é transforma palavras em representação numérica para que uma linguagem natural seja interpretada por um algoritmo.

Segundo RUSSELL; NORVIG (2013), há quatro estratégias para o estudo da IA: Pensando como um humano, Pensando racionalmente, Agindo como seres humanos e Agindo racionalmente. Para RUSSELL; NORVIG (2013) a categoria  “Agindo como seres humanos”, em uma abordagem do teste de Turing, o computador precisaria ter as seguintes capacidades:

    • processamento de linguagem natural para permitir que ele se comunique com sucesso em um idioma natural;
    • representação de conhecimento para armazenar o que sabe ou ouve;
    • raciocínio automatizado para usar as informações armazenadas com a finalidade de responder a perguntas e tirar novas conclusões;
    • aprendizado de máquina para se adaptar a novas circunstâncias e para detectar e extrapolar padrões.

A aprendizagem de máquina (ou machine learning) é um campo da inteligência artificial que se concentra no desenvolvimento de algoritmos e modelos que permitem que os computadores aprendam e tomem decisões ou façam previsões com base em dados, sem serem explicitamente programados.

Embora não foram descritas as outras três categorias, é nesta categoria que se encontram os conceitos e elementos para a construção deste trabalho.
Existem vários tipos de aprendizagem de máquina, incluindo aprendizagem supervisionada, aprendizagem não supervisionada e aprendizagem por reforço.
Na aprendizagem supervisionada, o algoritmo é treinado usando um conjunto de dados rotulados, onde cada exemplo de dados possui uma entrada e a saída desejada correspondente. O objetivo é aprender uma função que mapeie as entradas para as saídas corretas. Exemplos de algoritmos de aprendizagem supervisionada incluem regressão linear, árvores de decisão, redes neurais, entre outros.

Então, uma rede neural é composta por unidades interconectadas chamadas de neurônios artificiais, organizados em camadas. Cada neurônio recebe entradas, realiza um cálculo ponderado e aplica uma função de ativação para produzir uma saída. As conexões entre os neurônios são representadas por pesos, que são ajustados durante o treinamento do modelo.

[em construção...]


## METODOLOGIA

PyTorch é um framework de aprendizado de máquina e deep learning de código aberto amplamente utilizado. Ele foi desenvolvido pela equipe de pesquisa do Facebook AI e se tornou uma escolha popular entre pesquisadores e desenvolvedores devido à sua flexibilidade, facilidade de uso e eficiência computacional.

[em construção...]

## RESULTADOS ESPERADOS

código para treinar o modelo com textos jornalísticos

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Carregar o conjunto de dados
data = pd.read_csv("portuguese_bbc_text_cls.csv")

# Dividir os dados em treinamento e teste
train_data, test_data, train_labels, test_labels = train_test_split(data['text'], data['labels'], test_size=0.2, random_state=42)

# print('**>', train_labels.values)
# print('&&>', test_labels.values)
# print('##>', train_features)
# print('==>', test_features)

# Vetorização dos dados de texto
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Converter as classes de rótulo para valores numéricos
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Converter os dados em tensores do PyTorch
train_tensor = torch.tensor(train_features.toarray(), dtype=torch.float32)
test_tensor = torch.tensor(test_features.toarray(), dtype=torch.float32)

# Criar o tensor do PyTorch com os rótulos codificados
train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels_encoded, dtype=torch.long)

# Criar o conjunto de dados do PyTorch
train_dataset = TensorDataset(train_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_tensor, test_labels_tensor)

# Definir o modelo da rede neural
class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Definir os parâmetros do modelo
input_size = train_features.shape[1]
hidden_size = 128
num_classes = len(data['labels'].unique())

# Instanciar o modelo
model = TextClassifier(input_size, hidden_size, num_classes)

# Mover o modelo para a GPU, se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", device)
model.to(device)

# Mover os dados para a GPU
train_dataset.tensors = tuple(tensor.to(device) for tensor in train_dataset.tensors)
test_dataset.tensors = tuple(tensor.to(device) for tensor in test_dataset.tensors)

# Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Definir o tamanho do lote (batch size) e criar os dataloaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Treinamento do modelo
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data, batch_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")
    
# Avaliação do modelo
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_data, batch_labels in test_dataloader:
        outputs = model(batch_data)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test set: {accuracy*100:.2f}%")

# Salvar o modelo
torch.save(model, 'modelo.pth')

# Salvar o mapeamento de classes para valores numéricos
torch.save(label_encoder, "label_encoder.pth")
```

Código usando o modelo

```
import numpy as np

# Carregar o modelo treinado
model = torch.load("modelo.pth")

# Carregar o LabelEncoder
label_encoder = torch.load("label_encoder.pth")

# Definir o modo de avaliação
model.eval()

# Preparar o novo texto para classificação
# new_text = "vasco vai jogar hoje com o bangu"
# new_text = "venda de carro aumenta em 5%"
# new_text = "ministro da educação do governo promove envento na uff"
new_text = """
Após 6 anos, um dos melhores filmes de terror da Netflix ganhou 
prequel - e você já pode assistir aos dois
"""

new_features = vectorizer.transform([new_text])
new_tensor = torch.tensor(new_features.toarray(), dtype=torch.float32)

# Fazer a previsão com o modelo
with torch.no_grad():
    new_tensor = new_tensor.to(device)
    output = model(new_tensor)
    _, predicted = torch.max(output.data, 1)
    print('==>', predicted.item())
    # item = np.array([predicted.item()], dtype=np.int64)
    # print('##>', item)
    predicted_label = label_encoder.inverse_transform([predicted.item()])

print("Texto:", new_text)
print("Classe prevista:", predicted_label)
```

[em construção...]

## CONSIDERAÇÕES FINAIS

[em construção...]

## REFERÊNCIAS
[em construção...]
