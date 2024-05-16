#!/usr/bin/bash
#
#  Installs dependent packages into virtual environments that have already been set up as follows:
#  
#      pip install virtualenv
#      virtualenv -p python2 /scratch2/adriaan/venv-py2
#      virtualenv -p python3 /scratch2/adriaan/venv-py3
#
#  Some packages also require that github ssh keys have been activated.

# python3 environment
source ~/venv-py3/bin/activate
pip install --upgrade pip
pip cache purge
pip install ipykernel jupyter notebook
ipython3 kernel install --name "venv-py3" --user

pip install numpy scipy matplotlib
pip install git+https://github.com/ska-sa/{scape,katdal,katversion,katpoint,katsdpscripts,katsdpcal}
pip install pysolr paramiko
pip install git+https://github.com/telegraphic/PyGSM
py=`ls ~/venv-py3/lib/`
mv ~/venv-py3/lib/$py/site-packages/pygsm/gsm2016_components.h5 ~/venv-py3/lib/$py/site-packages/pygsm/gsm2016_components.h5~
wget -O i~/venv-py3/lib/$py/site-packages/pygsm/gsm2016_components.h5 https://zenodo.org/record/3479985/files/gsm2016_components.h5?download=1


# python2 environment
ipython3 kernel install --name "venv-py2" --user
source ~/venv-py2/bin/activate
pip install --upgrade pip
pip cache purge

pip install numpy scipy matplotlib
pip install git+https://github.com/ska-sa/katdal@python2

pip install git+https://github.com/scipy/weave@v0.16.0
pip install pygsheets==2.0.5 llvmlite==0.31.0 cityhash==0.3.0
pip install git+ssh://git@github.com/ska-sa/katholog@sdqm-072020
