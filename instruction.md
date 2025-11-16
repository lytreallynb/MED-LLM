# if python not found on mac
nano ~/.zshrc

(add the following to the last)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
(crtl + x, y, enter)

source ~/.zshrc
which pyenv

echo 'export PATH="$HOME/.pyenv/shims:$PATH"' >> ~/.zshrc
source ~/.zshrc
which python
which python3


# load the virtual environment
brew install pyenv
pyenv install 3.11.7

cd ~/Desktop/LLM
pyenv local 3.11.7 (best version fit for all libraries)

python --version
python3 --version

rm -rf venv (if you have a virtual environment already)
python -m venv venv
source venv/bin/activate
python --version


# install the required libraries
pip install --upgrade pip
pip install -r requirements.txt

# double check if download well
which python
which python3
echo $PATH

# download FDA data
mkdir -p ~/Desktop/LLM/data/raw
mkdir -p ~/Desktop/LLM/data/meta
nano download_fda_labels.py
source venv/bin/activate
python download_fda_labels.py
