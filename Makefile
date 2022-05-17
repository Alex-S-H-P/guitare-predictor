all: dataset/models/RTF.pickle
	@echo "\033[32;1mDone\033[0m"
dataset/models/RTF.pickle: dataset/musics/00_BN1-129-Eb_comp_hex.wav
	python3.10 -m codebase.musicHandler
dataset/music/00_BN1-129-Eb_comp_hex.wav:
	wget https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1
	unzip audio_hex-pickup_original.zip -d dataset/musics/
	rm audio_hex-pickup_original.zip