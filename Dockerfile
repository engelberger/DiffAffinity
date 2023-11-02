FROM us-docker.pkg.dev/colab-images/public/runtime
RUN /usr/bin/python3.10 -m pip -q install git+https://github.com/sokrypton/ColabDesign.git@gamma
RUN apt-get install aria2
RUN wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
RUN bash Mambaforge-Linux-x86_64.sh -bfp /usr/local
RUN mamba config --set auto_update_conda false
RUN mamba create -n DiffAffinity -y -c conda-forge python=3.9
#COPY . /tmp/DiffAffinity
WORKDIR /tmp/
RUN git clone -b dev https://github.com/engelberger/DiffAffinity.git
WORKDIR /tmp/DiffAffinity
RUN /usr/local/envs/DiffAffinity/bin/python -m pip install -r requirements.txt
RUN /usr/local/envs/DiffAffinity/bin/python -m pip install jaxlib==0.4.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN git clone https://github.com/engelberger/riemannian-score-sde.git
WORKDIR /tmp/DiffAffinity/riemannian-score-sde/
RUN git clone https://github.com/oxcsml/geomstats.git 
RUN /usr/local/envs/DiffAffinity/bin/python -m pip install -r requirements.txt
RUN /usr/local/envs/DiffAffinity/bin/python -m pip install -r requirements_exps.txt
RUN /usr/local/envs/DiffAffinity/bin/python -m pip install geomstats
WORKDIR /tmp/DiffAffinity/
RUN git clone https://github.com/oxcsml/geomstats.git 
RUN export GEOMSTATS_BACKEND=jax && /usr/local/envs/DiffAffinity/bin/python -m pip install -e geomstats