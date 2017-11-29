yes | conda env create -f thoughtvec_imagesearch_env.yml
source activate thoughtvec_imagesearch
python -m nltk.downloader -d /usr/local/share/nltk_data all
source deactivate
