#!/bin/bash

# 1. Identifica o executável do Python disponível
if command -v python3 &>/dev/null; then
    PYTHON_EXE=$(command -v python3)
elif command -v python &>/dev/null; then
    PYTHON_EXE=$(command -v python)
else
    echo "❌ Erro: Python não encontrado no sistema. Instale-o primeiro."
    exit 1
fi

echo "✅ Python encontrado em: $PYTHON_EXE"
echo "🐍 Versão: $($PYTHON_EXE --version)"

# 2. Define o nome da pasta da venv
VENV_NAME=".venv"

# 3. Cria a virtual env se ela não existir
if [ ! -d "$VENV_NAME" ]; then
    echo "⚙️ Criando ambiente virtual em '$VENV_NAME'..."
    $PYTHON_EXE -m venv "$VENV_NAME"
else
    echo "ℹ️ O ambiente virtual '$VENV_NAME' já existe."
fi

# 4. Ativa a venv
source "$VENV_NAME/bin/activate"

# 5. Instala as dependências se o requirements.txt existir
if [ -f "requirements.txt" ]; then
    echo "📦 Instalando dependências de requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✨ Tudo pronto! Ambiente configurado e bibliotecas instaladas."
else
    echo "⚠️ Aviso: 'requirements.txt' não encontrado. venv criada, mas nada foi instalado."
fi

# 6. Injeta a exportação do CUDA diretamente no ativador do ambiente (activate)
echo "🔧 Configurando carregamento automático do CUDA no terminal..."

cat << 'EOF' >> "$VENV_NAME/bin/activate"

# Exporta a raiz do projeto para o PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(dirname "$VIRTUAL_ENV")"

# CUDA/NVIDIA library path injection
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import os, glob; print(":".join(glob.glob(os.path.join(os.environ.get("VIRTUAL_ENV", ""), "lib", "python*", "site-packages", "nvidia", "*", "lib"))))')
EOF

echo "✅ Script activate modificado com sucesso para carregar GPUs NVIDIA."

exec bash --rcfile <(echo "source ~/.bashrc; source $VENV_NAME/bin/activate")