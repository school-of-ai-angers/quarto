FROM continuumio/miniconda3:4.7.10


WORKDIR /app

# Conda dependencies
COPY quarto/deep_submission/environment.yml .
RUN conda env create
RUN echo "conda activate quarto" >> ~/.bashrc
ENV PATH=/opt/conda/envs/quarto/bin:$PATH

# Main app
COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT [ "/opt/conda/envs/quarto/bin/python" ]

