#!/usr/bin/bash
#
#  Installs virtual environment and dependent packages, as required to work with
#  the DVS frameworks.
#  Ensure that your github ssh keys have been activated!
#

pip install virtualenv
virtualenv -p python3 ~/venv-py3

## IMPORTANT: always work in this python3 environment
source ~/venv-py3/bin/activate
pip install --upgrade pip
pip cache purge
pip install ipykernel jupyter notebook
ipython3 kernel install --name "python3" --user

pip install --upgrade nbdime
nbdime config-git --enable --global


pip install numpy scipy matplotlib allantools
pip install pysolr paramiko zernike

pip install git+https://github.com/ska-sa/{scape,katdal,katversion,katpoint,katsdpcal,katsdpcalproc}
# pip install git+ssh://git@github.com/ska-sa/dvsholog
# TODO: At present, the above repository is private - install as follows
pip install dvsholog

pip install git+https://github.com/telegraphic/PyGSM
PY=`ls ~/venv-py3/lib/`
mv ~/venv-py3/lib/$PY/site-packages/pygsm/gsm2016_components.h5 ~/venv-py3/lib/$PY/site-packages/pygsm/gsm2016_components.h5~
wget -O ~/venv-py3/lib/$PY/site-packages/pygsm/gsm2016_components.h5 https://zenodo.org/record/3479985/files/gsm2016_components.h5?download=1

pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


## Set up the DVS workspace
git clone https://github.com/ska-sa/dvs.git
git clone https://github.com/ska-sa/katsdpscripts.git
# git clone git@github.com:ska-sa/systems-analysis.git
# TODO: At present, the above package is private - a temporary situation. The workaround is the following:
gdown 1Ep3o1mElx2YsJM_B4Hh6j_LKU4mkFYpU && unzip systems-analysis-master.zip -d systems-analysis && rm systems-analysis-master.zip
ln -f -s `pwd`/systems-analysis/analysis dvs/libraries/analysis
ln -f -s `pwd`/katsdpscripts/katsdpscripts dvs/libraries/katsdpscripts

ipython3 kernel install --name "dvs" --user
# Add a custom pythonpath
KS=`jupyter kernelspec list | sed -n -E 's/\s*(dvs)\s+(\S+)/\2/p'`/kernel.json
PK=\"env\":{\"PYTHONPATH\":\"`pwd`/dvs\"}
sed -i -E ':a; N; $! ba; s@(.*)}@\1,'$PK'\n}@' $KS


# Install rclone - that's if you want to download local cached datasets! See https://rclone.org/install
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
chmod +x `ls -d rclone-*-linux-amd64`/rclone
cp `ls -d rclone-*-linux-amd64`/rclone /home/aph/venv-py3/bin
rm -rf rclone-*-linux-amd64*

echo You may start the notebook server by using the following command:
echo `readlink -f dvs/startjnb` 