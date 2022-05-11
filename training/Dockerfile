# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM tensorflow/tensorflow
RUN apt-get update
RUN apt-get install -y wget
RUN groupadd -r train && useradd -m --no-log-init -r -g train train

RUN mkdir -p /opt/train /input /output \
    && chown train:train /opt/train /input /output

USER train

WORKDIR /opt/train

ENV PATH="/home/train/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=train:train requirements.txt /opt/train/
RUN python -m pip install --user -r requirements.txt

COPY --chown=train:train config.py /opt/train/
COPY --chown=train:train ctdataset.py /opt/train/
COPY --chown=train:train main.py /opt/train/
COPY --chown=train:train metrics.py /opt/train/
COPY --chown=train:train train.py /opt/train/
COPY --chown=train:train algorithm/ /opt/train/algorithm/
COPY --chown=train:train config/ /opt/train/config/

RUN wget https://github.com/parkerhuynh/STOIC-Challenge-2021/releases/download/v0.1.0/densenet201_proSevere.h5 -O /opt/train/algorithm/densenet201_proSevere.h5
RUN wget https://github.com/parkerhuynh/STOIC-Challenge-2021/releases/download/v0.1.0/densenet169_proSevere.h5 -O /opt/train/algorithm/densenet169_proSevere.h5
RUN wget https://github.com/parkerhuynh/STOIC-Challenge-2021/releases/download/v0.1.0/densenet121_proSevere.h5 -O /opt/train/algorithm/densenet121_proSevere.h5

RUN wget https://github.com/parkerhuynh/STOIC-Challenge-2021/releases/download/v0.1.0/densenet201_proCOVID.h5 -O /opt/train/algorithm/densenet201_proCOVID.h5
RUN wget https://github.com/parkerhuynh/STOIC-Challenge-2021/releases/download/v0.1.0/densenet169_proCOVID.h5 -O /opt/train/algorithm/densenet169_proCOVID.h5
RUN wget https://github.com/parkerhuynh/STOIC-Challenge-2021/releases/download/v0.1.0/densenet121_proCOVID.h5 -O /opt/train/algorithm/densenet121_proCOVID.h5

ENTRYPOINT python -m main $0 $@
