import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_model import MobileNetV2FER # Importa o novo modelo

def main():
    # Carrega os DataLoaders
    train_loader = torch.load("train_loader.pth", weights_only=False)
    val_loader = torch.load("val_loader.pth", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Instancia o modelo, função de perda e otimizador
    model = MobileNetV2FER(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Número de épocas
    num_epochs = 50  

    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(outputs, labels) # Calcula perda do modelo
            loss.backward() # Calcula os gradientes com base na perda (backpropagation)
            optimizer.step() # Atualiza os valores dos pesos

            running_loss += loss.item()

        print(f"Época {epoch+1}, Perda de Treinamento: {running_loss/len(train_loader):.4f}")

        # Avaliação no conjunto de validação
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Recebe o índice da classe com maior probabilidade segundo o modelo
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Acurácia de Validação: {accuracy:.2f}%")

    # Salva o modelo treinado
    torch.save(model.state_dict(), "facial_expression_mobilenetv2.pth")
    print("Treinamento concluído e modelo salvo.")

if __name__ == "__main__":
    main()


