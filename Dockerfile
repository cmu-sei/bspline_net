# bspline_net (Github)

# Copyright 2026 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

# This Software includes and/or makes use of Third-Party Software each subject to its own license.

# DM26-0750

FROM nvcr.io/nvidia/pytorch:23.08-py3

# ============ CERTIFICATES & PROXY ============
# if you need to add a custom certificate, uncomment the following lines and add your certificate to the Dockerfile directory
#COPY CERT.crt /usr/local/share/ca-certificates/
#RUN update-ca-certificates && chmod 644 /etc/ssl/certs/ca-certificates.crt

#
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    VCPKG_FORCE_SYSTEM_BINARIES=1

# If you build behind a proxy, uncomment and fill in:
# ENV HTTP_PROXY="your_proxy.com:80" \
#     HTTPS_PROXY="your_proxy.com:80" \
#     NO_PROXY="localhost,.local" \
#     http_proxy="your_proxy.com:80" \
#     https_proxy="your_proxy.com:80" \
#     no_proxy="localhost,.local"

# ============ SYSTEM PACKAGES ============
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1-mesa-glx figlet sudo tzdata tmux vim software-properties-common \
        texlive dvipng texlive-latex-extra texlive-fonts-recommended cm-super \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============ USER SETUP ============
RUN useradd -l -ms /bin/bash hspn-user \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && adduser hspn-user sudo

ARG BSPLINENETHOME=/home/hspn-user
WORKDIR ${BSPLINENETHOME}
USER hspn-user
ENV PATH=/home/hspn-user/.local/bin:/usr/local/bin:${PATH}

# ============ APPLICATION ============
COPY --chown=hspn-user:hspn-user ./docker_scripts ./docker_scripts
COPY --chown=hspn-user:hspn-user ./docker_scripts/bash_aliases ${BSPLINENETHOME}/.bash_aliases
COPY --chown=hspn-user:hspn-user pyproject.toml .
COPY --chown=hspn-user:hspn-user ./src ./src
COPY --chown=hspn-user:hspn-user ./tests ./tests

ARG DEV=false
RUN if [ "$DEV" = "true" ]; then \
        python3 -m pip install --no-cache-dir -e ".[dev]"; \
    else \
        python3 -m pip install --no-cache-dir -e "."; \
    fi \
    && python3 -m pip install --no-cache-dir jupyter

# ============ ENTRYPOINT ============
USER root
ENTRYPOINT ["./docker_scripts/entrypoint.sh"]