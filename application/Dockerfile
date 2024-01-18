FROM python:3.11-slim-bullseye as builder
# Tiktoken requires Rust toolchain, so build it in a separate stage
RUN apt-get update && apt-get install -y gcc curl g++
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && apt-get install --reinstall libc6-dev -y
ENV PATH="/root/.cargo/bin:${PATH}"

ENV CC=gcc CXX=g++
RUN echo "Detecting platform..." 
RUN export PLATFORM=$(uname -m | sed 's/\(x86_64\)/amd64/; s/(armv7l|aarch64)\//arm\//') 
RUN echo "Platform detected: ${PLATFORM}" 

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install -y wget unzip
RUN wget https://d3dg1063dc54p9.cloudfront.net/models/embeddings/mpnet-base-v2.zip
RUN mv mpnet-base-v2.zip all-mpnet-base-v2.zip
RUN unzip all-mpnet-base-v2.zip -d model
RUN rm all-mpnet-base-v2.zip
RUN pip install --no-cache-dir -vvv cmake==3.22.1 llama-cpp-python

FROM python:3.11-slim-bullseye

# Copy pre-built packages and binaries from builder stage
COPY --from=builder /usr/local/ /usr/local/

WORKDIR /app
COPY --from=builder /model /app/model

COPY . /app/application
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=true

EXPOSE 7091

CMD ["gunicorn", "-w", "2", "--timeout", "120", "--bind", "0.0.0.0:7091", "application.wsgi:app"]
