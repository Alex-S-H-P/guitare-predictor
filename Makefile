all: requirements.txt dataset/models/RTF.pickle
	@echo "\033[32;1mDone\033[0m"
	bash app.sh
dataset/models/RTF.pickle dataset/models/EMBED.pickle: dataset/musics/00_BN1-129-Eb_comp_hex.wav
	python3.10 -m codebase.musicHandler
dataset/music/00_BN1-129-Eb_comp_hex.wav:
	wget https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1 -O audio_hex-pickup_original.zip
	unzip audio_hex-pickup_original.zip -d dataset/musics/
	rm audio_hex-pickup_original.zip
requirements.txt: codebase/__init__.py codebase/classify.py codebase/interpreter.py codebase/librosaTest.py codebase/musicHandler.py codebase/path_handler.py codebase/utillib/
	bash ./freeze.sh