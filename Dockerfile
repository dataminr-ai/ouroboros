FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu20.04

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

## Required packages
RUN apt update
RUN apt-get install -y curl wget git-core gcc make zlib1g zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev libffi-dev lzma liblzma-dev libbz2-dev

## gh
RUN mkdir -p -m 755 /etc/apt/keyrings
RUN wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
RUN chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
RUN apt update && apt-get install -y gh

## Python
RUN curl https://pyenv.run | bash

ENV HOME "/root"
ENV PYENV_ROOT "$HOME/.pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

RUN pyenv install 3.10.13
RUN pyenv global 3.10.13

ADD src /opt/src
ADD pyproject.toml /opt/
WORKDIR /opt/

## Setup env
RUN pip install build wheel torch==2.3.1
RUN pip install -e ".[fast]"

ENTRYPOINT ["/bin/bash"]