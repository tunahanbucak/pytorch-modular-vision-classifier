import torch
from torch import nn

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    # Modeli eğitim moduna al
    model.train()

    # Kümülatif kayıp ve doğruluk sayaçlarını başlat
    train_loss, train_acc = 0, 0

    # Mini-batch'ler üzerinde döngü
    for batch, (X, y) in enumerate(dataloader):
        # 1. İleri Yayılım (Forward Pass)
        y_pred = model(X)

        # 2. Kaybı hesapla ve biriktir
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Gradyanları sıfırla
        optimizer.zero_grad()

        # 4. Geriye yayılım (Backpropagation)
        loss.backward()

        # 5. Ağırlıkları güncelle
        optimizer.step()

        # Doğruluk metriğini hesapla
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Batch başına ortalama kayıp ve doğruluğu hesapla
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
):
    # Modeli değerlendirme moduna al
    model.eval()

    # Test kayıp ve doğruluk sayaçlarını başlat
    test_loss, test_acc = 0, 0

    # Gradyan takibini kapat (bellek tasarrufu)
    with torch.inference_mode():
        # Test batch'leri üzerinde döngü
        for batch, (X, y) in enumerate(dataloader):
            # 1. İleri Yayılım (Forward Pass)
            test_pred_logits = model(X)

            # 2. Test kaybını hesapla
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Test doğruluğunu hesapla
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Batch başına ortalama kayıp ve doğruluğu hesapla
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# train_step ve test_step'i birleştiren ana eğitim fonksiyonu
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
):

    # Metrik sonuçları için boş sözlük
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Belirtilen epoch sayısı kadar eğitim ve test yap
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn
        )

        # Epoch sonuçlarını ekrana yazdır
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Sonuç sözlüğünü güncelle
        # Değerleri float olarak sözlüğe kaydet
        results["train_loss"].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        results["train_acc"].append(
            train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        )
        results["test_loss"].append(
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )
        results["test_acc"].append(
            test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc
        )

    # Sonuç sözlüğünü döndür
    return results
