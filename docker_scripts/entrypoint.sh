#!/bin/bash

# check that values present for user and group id
if [ "${MY_UID}" -eq "0" ] || [ -z "$MY_GID" ] ; then
    echo "Add -e MY_UID=\$(id -u) -e MY_GID=\$(id -g) to your docker run command."
    exit 1
fi
echo "Changing UID and GID for hspn-user to host values."
# temporary relocation of home directory so that usermod doesn't take forever
usermod -d /tmp/home/hspn-user hspn-user
usermod -u ${MY_UID} hspn-user
groupmod -g ${MY_GID} hspn-user
usermod -d /home/hspn-user hspn-user
usermod -aG sudo hspn-user
echo "Changing ownership. This can take a long time."

chown -R ${MY_UID}:${MY_GID}  /home/hspn-user/
#chmod 644 ~/.bashrc
ls -la /home/hspn-user
# If you need to chown more directories, but ignoring certain paths, do
# something along these lines.
#  HOME=/home/hspn-user
#  # change the owner of the home directory
#  chown hspn-user:hspn-user ${HOME}
#  # recursively change the owner of subdirectories excluding certain ones to
#  # save A LOT of time. The first -path ${HOME} is important for chown -R.
#  find ${HOME} -maxdepth 1 \( \
#      -path ${HOME} \
#      -o -path ${HOME}/ignore1 -prune \
#      -o -path ${HOME}/ignore2 -prune \
#      -o -path ${HOME}/ignore3 -prune \) \
#      -o -exec chown -R hspn-user:hspn-user {} +

# change user
echo "Changing to hspn-user."
su hspn-user
