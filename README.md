# Redes neurais para classificação de textos jornalísticos e processamento paralelo em CUDA e PyTorch

### André Ricardo Prazeres Rodrigues

## INTRODUÇÃO

Desenvolver um sistema inteligente capaz de classificar automaticamente textos jornalísticos com base em suas características semânticas e epistêmicas e, em seguida, gerar resumos relevantes dessas notícias.
Isso envolve o uso de lógica epistêmica para modelar conhecimento e crenças em relação às notícias, redes neurais para análise semântica e processamento paralelo em CUDA para acelerar o processamento.


## METODOLOGIA

PyTorch é um framework de aprendizado de máquina e deep learning de código aberto amplamente utilizado. Ele foi desenvolvido pela equipe de pesquisa do Facebook AI e se tornou uma escolha popular entre pesquisadores e desenvolvedores devido à sua flexibilidade, facilidade de uso e eficiência computacional.

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

## CONSIDERAÇÕES FINAIS

## REFERÊNCIAS
