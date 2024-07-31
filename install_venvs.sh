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


pip install numpy scipy matplotlib
pip install pysolr paramiko zernike

pip install git+https://github.com/ska-sa/{scape,katdal,katversion,katpoint,katsdpscripts,katsdpcal}
pip install git+ssh://git@github.com/ska-sa/dvsholog
# TODO: At present, the above package is private - a temporary situation. The workaround is to obtain a "zip" package and do:
# pip install dvsholog-main.zip

pip install git+https://github.com/telegraphic/PyGSM
py=`ls ~/venv-py3/lib/`
mv ~/venv-py3/lib/$py/site-packages/pygsm/gsm2016_components.h5 ~/venv-py3/lib/$py/site-packages/pygsm/gsm2016_components.h5~
wget -O i~/venv-py3/lib/$py/site-packages/pygsm/gsm2016_components.h5 https://zenodo.org/record/3479985/files/gsm2016_components.h5?download=1


## Set up the DVS workspace
git clone git@github.com:ska-sa/dvs.git
git clone git@github.com:ska-sa/systems-analysis.git
# TODO: At present, the above package is private - a temporary situation. The workaround is to obtain a "zip" package and do:
# unzip systems-analysis-master.zip -d systems-analysis
ln -s `pwd`/systems-analysis/analysis dvs/libraries/analysis

ipython3 kernel install --name "dvs" --user
jupyter kernelspec list | grep dvs
echo ACTION: Insert the following line into the above dvs/kernel.json 
echo  \"env\": {\"PYTHONPATH\":\"`pwd`/dvs\"},
