"""
Istruzioni rapide per eseguire questo script
===========================================
1) Crea e attiva un ambiente virtuale (consigliato):
   python -m venv .venv
   .venv\\Scripts\\activate

2) Installa le dipendenze:
   pip install numpy matplotlib

3) Scarica il dataset EMNIST (formato IDX):
   - Vai su: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
   - Estrai l'archivio.
   - Dalla cartella estratta, prendi questi 4 file:
     emnist-letters-train-images-idx3-ubyte
     emnist-letters-train-labels-idx1-ubyte
     emnist-letters-test-images-idx3-ubyte
     emnist-letters-test-labels-idx1-ubyte
   - Mettili nella stessa cartella di questo script.

4) Esegui:
   python emnist_svd.py
"""

import numpy as np
import matplotlib.pyplot as plt 

from pathlib import Path
##STRUTTURA DEL DATASET
#IMMAGINI
# X_train.shape → (112800, 784)

# riga 0:     [0, 0, 255, 128, ...]   ← 784 pixel dell'immagine 0
# riga 1:     [0, 12, 200, 0,  ...]   ← 784 pixel dell'immagine 1
# riga 2:     [255, 0, 0, 43,  ...]   ← 784 pixel dell'immagine 2
# ...
# riga 112799 [...]                   ← 784 pixel dell'immagine 112799

#ETICHETTE(se per ogni lettera ho 2400 foto, in y_train avrò 2400 elementi con etichetta di quella lettera(es:supponendo siano ordinate le prime 2400 etichette saranno "a"))
# y_train.shape → (112800,)

# posizione 0:      'a'   ← etichetta dell'immagine 0
# posizione 1:      'a'   ← etichetta dell'immagine 1
# posizione 2:      'b'   ← etichetta dell'immagine 2
# ...
# posizione 112799: 'z'   ← etichetta dell'immagine 112799

#%% Funzione per leggere i file EMNIST in formato binario
#funzione per leggere tutte le immagini, partendo dall'header che da le info come magic number(per capire se il file contiene foto dei char o etichette), n di immagini , righe e colonne di ogni foto(28 * 28) 
#le ultime 2 righe permettono di creare un array di byte numpy(ogni elemento è un pixel di una foto) e di riorganizzarli in una matrice che ha una righa per ogni immagine con tutti i pixel di quest'ultima
#la funzione restituisce una matrice di di immagini di lettere , ogni riga è una serie di pixel che rappresenta un'immagine(la funzione vettorizza già le immagini(?))
def leggi_immagini(filepath):
    with open(filepath, 'rb') as f:
        # I primi 16 byte sono header: magic number, n immagini, righe, colonne
        magic   = int.from_bytes(f.read(4), 'big')
        n_img   = int.from_bytes(f.read(4), 'big')
        n_righe = int.from_bytes(f.read(4), 'big')
        n_col   = int.from_bytes(f.read(4), 'big')
        # Il resto sono i pixel, un byte per pixel
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_img, n_righe * n_col)
    return data
#funzione per leggere i file con le etichette del dataset , legge magic number e numero di etichette(una per ogni lettera quindi 26 in questo caso)
#poi legge i byte per ogni etichetta e le mette in un array np in cui in ogni casella c'è una lettera(ogni immagine di una lettera ha un etichetta, x e y sono "allineate")
#la funzione restituisce un'array di etichette
def leggi_etichette(filepath):
    with open(filepath, 'rb') as f:
        # I primi 8 byte sono header: magic number, n etichette
        magic  = int.from_bytes(f.read(4), 'big')
        n_lab  = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

#%% Caricamento dataset
print('Caricamento dataset EMNIST Letters...')
BASE_DIR = Path(__file__).resolve().parent
X_train = leggi_immagini(BASE_DIR / 'emnist-letters-train-images-idx3-ubyte')
y_train = leggi_etichette(BASE_DIR / 'emnist-letters-train-labels-idx1-ubyte')
X_test  = leggi_immagini(BASE_DIR / 'emnist-letters-test-images-idx3-ubyte')
y_test  = leggi_etichette(BASE_DIR / 'emnist-letters-test-labels-idx1-ubyte')

# Le etichette EMNIST Letters vanno da 1 a 26 (1=a, 2=b, ...)
# Convertiamo in lettere
y_train = np.array([chr(ord('a') + l - 1) for l in y_train])#converte in lettera ogni etichetta per ogni foto di lettera
y_test  = np.array([chr(ord('a') + l - 1) for l in y_test])

print('Training: %d immagini' % X_train.shape[0])#n righe della matrice
print('ogni immagine ha %d pixel' % X_train.shape[1])#colonne della matrice
print('Test:     %d immagini' % X_test.shape[0])
print('ogni immagine ha %d pixel' % X_test.shape[1])
print('Classi:', np.unique(y_train))

#%% Costruzione matrice dataset A (784 x N)
# Trasponiamo X per avere ogni immagine su una colonna
# esattamente come nel codice eigenfaces: A[:,i] = vettore immagine

nn, mm = 28, 28          # dimensioni immagine
n_pixel = nn * mm        # 784

# Usiamo un sottoinsieme per iniziare (es. 5000 immagini)
#%% Costruzione matrici trial e test
#prendo x foto per ogni lettera 
x_train_per_letter = 300 
x_test_per_letter  = 80
rng = np.random.default_rng(67) #generatore di numeri casuali con seed 67(numero a caso)

#per ogni carattere(for c in y_train) , prendimi x_train_per_letter indici delle foto(prendo il minimo tra quelle che dico io e quelle presenti per carattere, nel caso in cui quelle presenti fossero minori,non questo)
#vengono presi con choice x_train_per_letter indici di foto per ogni lettera 
#concatenate crea un array numpy quindi un array di x_train_per_letter * ogni lettera(26)(ogni elemento è l'indirizzo di un immagine)
idx_train = np.concatenate([
    rng.choice(np.where(y_train == c)[0], size=min(x_train_per_letter, np.sum(y_train == c)), replace=False)
    for c in np.unique(y_train)
])

idx_test = np.concatenate([
    rng.choice(np.where(y_test == c)[0], size=min(x_test_per_letter, np.sum(y_test == c)), replace=False)
    for c in np.unique(y_test)
])
print(f"contenuto di idx_test:{idx_test} ")

N_train = idx_train.shape[0]#prendo la lunghezza dell'array numpy
N_test  = idx_test.shape[0]

A_trial      = X_train[idx_train, :].T.astype(float)  # matrice di train (784 x N_train), NE FA LA TRASPOSTA(.T) cosicchè io abbia una matrice in cui ogni colonna è la sequenza di pixel di un'immmagine
A_test       = X_test[idx_test, :].T.astype(float)     # (784 x N_test), stesso discorso per la trasposta ,inoltre i pixel interi vengono convertiti in float per avere i calcoli della SVD
labels_trial = y_train[idx_train]
labels_test  = y_test[idx_test]

# Calcolo e sottrazione della lettera media (calcolata sul training)
lettera_media = np.mean(A_trial, axis=1) 

plt.figure(1)
plt.imshow(lettera_media.reshape(nn, mm), cmap='gray')
plt.axis('off')
plt.title('Lettera media del dataset')
plt.show()

# Sottraiamo la media da trial e test
for i in range(N_train):
    A_trial[:, i] = A_trial[:, i] - lettera_media
for i in range(N_test):
    A_test[:, i] = A_test[:, i] - lettera_media

print('A_trial: %d x %d' % A_trial.shape)
print('A_test:  %d x %d' % A_test.shape)
print('Normalizzazione completata')


#%% Calcolo SVD della matrice di trial (forma economy)
#A avrà 784 righe(n pixel per immagini) e N immagini(in base al numero di immagini che prendo)
print('Calcolo SVD...')
# calcolo la SVD della matrice di train , uso la forma economy con full_matrices = false
U, S, Vt = np.linalg.svd(A_trial, full_matrices=False)
print('SVD completata')
print('U: %d x %d' % U.shape)#dimensione di U che sarà 784 x 784
print('S: %d valori singolari' % S.shape[0])#784 valori singolari, quelli dopo daranno contributo 0 e vengono troncati dalla economy
print('Vt: %d x %d' % Vt.shape)#


#%% Energia cumulativa per scegliere K(abbastanza K da essere vicini a 1)
E = np.cumsum(S**2) / np.sum(S**2)
#grafico per mostrare con quanti valori singolari si raggiunge un buon valore per l'energia cumulativa
plt.figure(2)
plt.plot(E, 'b.', markersize=3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% energia')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% energia')
plt.xlabel('k')
plt.ylabel('Energia cumulativa')
plt.title('Energia cumulativa dei valori singolari')
plt.legend()
plt.grid(True)
plt.show()

# Troviamo il K minimo per conservare il 95% dell'energia
K_95 = np.argmax(E >= 0.95) + 1
K_99 = np.argmax(E >= 0.99) + 1
print('K per 95%% di energia: %d' % K_95)
print('K per 99%% di energia: %d' % K_99)

#%% Scelta di K e proiezione nello spazio ridotto
K = K_95                    
UK = U[:, 0:K]              # prende le prime prime K eigenletters(colonne di U ) (784 x K)

# Proiettiamo tutte le immagini di trial
P_trial = UK.T @ A_trial    # (K x N_trial), creo la matrice P_trial, in cui ogni colonna pj da le info per il carattere aj
print('Spazio proiettato trial: %d x %d' % P_trial.shape)
#per l'immagine scelta a caso dal dataset calcolero il suo p e lo confronterò con tutti i pj di P per associarlo alla lettera corretta
#%% Classificazione di un'immagine di test
# scegliamo una lettera casuale tra tutte le classi disponibili
lettera_scelta = np.random.choice(np.unique(labels_test))
# prendiamo un'immagine casuale di quella classe, prima cerco gli indici delle immagini di quella lettera e poi la seleziono
indici_classe = np.where(labels_test == lettera_scelta)[0]
j_ind = np.random.choice(indici_classe)

lettera_test = A_test[:, j_ind]
etichetta_vera = labels_test[j_ind]

# Proiettiamo la lettera test
p_test = UK.T @ lettera_test   # vettore di K elementi(@ = prodotto matriciale, ricorda!)

# Visualizziamo la lettera da riconoscere
plt.figure(3)
plt.imshow(lettera_test.reshape(nn, mm), cmap='gray')
plt.axis('off')
plt.title('Lettera da riconoscere (etichetta vera: %s)' % etichetta_vera)
plt.show()

#%% Calcolo distanze e identificazione
dist_p = np.zeros(P_trial.shape[1])
for j in range(P_trial.shape[1]):
    dist_p[j] = np.linalg.norm(P_trial[:, j] - p_test)

# Creiamo etichette per l'asse x
x_ticks  = []
x_labels = []
for i, lettera in enumerate(np.unique(labels_trial)):
    centro = i * x_train_per_letter + x_train_per_letter // 2
    x_ticks.append(centro)
    x_labels.append(lettera)

plt.figure(4)
plt.plot(dist_p, 'b.', markersize=2, label='distanze')
plt.xticks(x_ticks, x_labels)
plt.xlabel('lettera')
plt.title('Distanze dalla lettera test (etichetta vera: %s)' % etichetta_vera)
plt.legend()
plt.show()

# Lettera con distanza minima
js = np.argmin(dist_p)
etichetta_trovata = labels_trial[js]

print('Etichetta vera:    %s' % etichetta_vera)
print('Etichetta trovata: %s' % etichetta_trovata)
print('Risultato: %s' % ('CORRETTO' if etichetta_vera == etichetta_trovata else 'SBAGLIATO'))

# Visualizziamo la lettera trovata
lettera_trovata = A_trial[:, js]
plt.figure(5)
plt.imshow(lettera_trovata.reshape(nn, mm), cmap='gray')
plt.axis('off')
plt.title('Lettera trovata: %s' % etichetta_trovata)
plt.show()

#%% Valutazione completa su tutto il test set e matrice di confusione

# Classifichiamo tutte le immagini di test 

print('Classificazione di tutto il test set...')
classi = np.unique(labels_trial)
n_classi = len(classi)

# dizionario per convertire lettera -> indice numerico  {'a':0, 'b':1, ..., 'z':25}
lettera_to_idx = {c: i for i, c in enumerate(classi)}

predizioni  = []
vere        = []
#per tutte le immagini di test faccio ciò che facevo prima con una lettera a caso
for j in range(A_test.shape[1]):
    # proietta l'immagine di test
    p_t = UK.T @ A_test[:, j]
    # calcola distanze da tutti i trial
    dist = np.array([np.linalg.norm(P_trial[:, k] - p_t) 
                     for k in range(P_trial.shape[1])])
    # prendi la classe con distanza minima
    js_best = np.argmin(dist)
    predizioni.append(labels_trial[js_best])
    vere.append(labels_test[j])
#vere contiene le etichette reali di ogni immagine di test
#predizioni contiene le etichette trovate calcolando la distanza minima dalle p 

predizioni = np.array(predizioni)
vere       = np.array(vere)

# Accuratezza globale
#calcolo la media delle predizioni azzeccate (cioè n predizioni azzeccate(dove predizioni == vere)/n predizioni totali )
accuracy = np.mean(predizioni == vere) * 100
print('Accuratezza globale: %.2f%%' % accuracy)

# Costruzione matrice di confusione
conf_matrix = np.zeros((n_classi, n_classi), dtype=int)
for v, p in zip(vere, predizioni):
    conf_matrix[lettera_to_idx[v], lettera_to_idx[p]] += 1

# Visualizzazione matrice di confusione
plt.figure(6, figsize=(14, 12))
plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xticks(range(n_classi), classi, fontsize=8)
plt.yticks(range(n_classi), classi, fontsize=8)
plt.xlabel('Lettera predetta')
plt.ylabel('Lettera vera')
plt.title('Matrice di confusione (K=%d, acc=%.1f%%)' % (K, accuracy))

# Scrivi i numeri nelle celle più significative
for i in range(n_classi):
    for j in range(n_classi):
        if conf_matrix[i, j] > 0:
            plt.text(j, i, str(conf_matrix[i, j]),
                     ha='center', va='center', fontsize=5,
                     color='white' if conf_matrix[i, j] > conf_matrix.max()*0.5 
                     else 'black')

plt.tight_layout()
plt.show()

#%% Accuratezza per lettera - grafico a barre
accuratezze_per_lettera = []
for i, c in enumerate(classi):
    totale   = conf_matrix[i, :].sum()
    corretti = conf_matrix[i, i]
    acc = 100 * corretti / totale if totale > 0 else 0
    accuratezze_per_lettera.append(acc)
    print('  %s: %d/%d = %.1f%%' % (c, corretti, totale, acc))

plt.figure(8, figsize=(16, 6))
bars = plt.bar(classi, accuratezze_per_lettera, color='steelblue', edgecolor='black', linewidth=0.5)

# linea dell'accuratezza media globale
plt.axhline(y=accuracy, color='red', linestyle='--', linewidth=1.5,
            label='Accuratezza globale: %.1f%%' % accuracy)

# scrive il valore sopra ogni barra
for bar, acc in zip(bars, accuratezze_per_lettera):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             '%.0f%%' % acc, ha='center', va='bottom', fontsize=7)

plt.ylim(0, 110)
plt.xlabel('Lettera')
plt.ylabel('Accuratezza (%)')
plt.title('Accuratezza per lettera - SVD (K=%d, acc. globale=%.1f%%)' % (K, accuracy))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
    
    #%% Accuratezza al variare di K
# valori_K = [10, 20, 50, 100, K_95, K_99, 300]
# accuratezze = []

# for k in valori_K:
#     Uk_temp = U[:, 0:k]
#     P_temp  = Uk_temp.T @ A_trial
    
#     corretti = 0
#     for j in range(A_test.shape[1]):
#         p_t  = Uk_temp.T @ A_test[:, j]
#         dist = np.array([np.linalg.norm(P_temp[:, col] - p_t)
#                          for col in range(P_temp.shape[1])])
#         pred = labels_trial[np.argmin(dist)]
#         if pred == labels_test[j]:
#             corretti += 1
    
#     acc = corretti / A_test.shape[1] * 100
#     accuratezze.append(acc)
#     print('K=%d → accuratezza=%.2f%%' % (k, acc))

# plt.figure(7)
# plt.plot(valori_K, accuratezze, 'bo-')
# plt.axvline(x=K_95, color='r', linestyle='--', label='K_95')
# plt.axvline(x=K_99, color='g', linestyle='--', label='K_99')
# plt.xlabel('K (numero di eigenletters)')
# plt.ylabel('Accuratezza (%)')
# plt.title('Accuratezza al variare di K')
# plt.legend()
# plt.grid(True)
# plt.show()
