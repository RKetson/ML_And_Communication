import pandas as pd

# Converte os pontos gerados pela ferramenta AFF3CT em dicionários BER e SER
# Prestar atenção ao parâmetro meta, valor deve ser verdadeiro quando for inserido os metadados no txt
def txt_to_dict(local, meta=True):
    if meta:
        skip = 5
    else:
        skip = None
    
    df = pd.read_csv(local, skiprows=skip, comment="#", delimiter="|", header=None, keep_default_na=False, usecols=[1,6,7], names=["Eb/N0 (dB)","BER","SER"], index_col=0)

    return df["BER"].to_dict(), df["SER"].to_dict()