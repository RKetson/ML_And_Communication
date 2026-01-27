# 🚀 Machine Learning and Communication

Este repositório contém o ambiente automatizado e a estrutura de pastas necessária para o desenvolvimento, treinamento e validação de modelos de ML/DL e curvas de performance.

## 📁 Estrutura do Repositório

A organização do projeto segue uma lógica de validação de dados:

* **`Pontos/`**: Armazena dados consolidados e validados. Aqui ficam os pontos de curvas de erros gerados por ferramentas externas ou scripts de ML que já passaram pelo processo de validação.
* **`Modelos/`**: Armazena os pesos de redes neurais já treinadas e validadas, e pontos de constelações validados.
* **`Buffer/`**: Área de rascunho e testes temporários. Contém curvas não validadas, diagramas de constelação experimentais ou arquivos de pesos de redes neurais (`.h5`, `.pt`, `.ckpt`) que ainda não foram homologados.
* **`setup_venv.sh`**: Script de automação para configuração do ambiente Python.
* **`requirements.txt`**: Lista de dependências e bibliotecas necessárias para o projeto.

---

## 🛠️ Configuração do Ambiente

Para garantir que todos os colaboradores utilizem as mesmas versões de bibliotecas, utilizamos um script Bash que automatiza a criação do Ambiente Virtual (venv).

### Pré-requisitos

* Sistema operacional Linux (ou WSL no Windows).
* Python 3 instalado.

### Passo a Passo

1. **Dar permissão de execução ao script:**
Na primeira vez que baixar o repositório, execute:
```bash
chmod +x setup_venv.sh

```


2. **Executar o instalador:**
```bash
./setup_venv.sh

```


O script irá identificar automaticamente o seu Python, criar a pasta `.venv` e instalar todas as dependências do `requirements.txt`.


3. **Ativar o ambiente:**
Após a execução do script, ative o ambiente no seu terminal atual:
```bash
source .venv/bin/activate

```



---

## 🧪 Fluxo de Trabalho Sugerido

1. Desenvolva seu script e gere os resultados iniciais na pasta **`Buffer/`**.
2. Após verificar que os dados (curvas, constelações ou pesos) estão corretos e estáveis, mova os resultados definitivos para a pasta **`Modelos/`**.
3. Nunca envie a pasta `.venv/` para o controle de versão (ela já está listada no `.gitignore`).

---

### Continuidade do Projeto

Se você adicionar uma biblioteca nova ao projeto enquanto trabalha, não esqueça de atualizar o arquivo de requisitos:

```bash
pip freeze > requirements.txt

```
