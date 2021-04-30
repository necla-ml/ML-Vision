.PHONY: clone checkout co pull 
.PHONY: build install uninstall clean

HOST:=$(shell uname -s | tr A-Z a-z)
CHANNEL?=NECLA-ML

all: build

## Environment

conda-install:
	wget -O $(HOME)/Downloads/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	sh $(HOME)/Downloads/Miniconda3-latest-Linux-x86_64.sh -b -p $(HOME)/miniconda3
	rm -fr $(HOME)/Downloads/Miniconda3-latest-Linux-x86_64.sh

conda-env: $(HOME)/miniconda3
	eval "`$(HOME)/miniconda3/bin/conda shell.bash hook`" && conda env create -n $(ENV) -f $(ENV).yml

conda-setup: $(HOME)/miniconda3
	echo '' >> $(HOME)/.bashrc
	echo 'eval "`$$HOME/miniconda3/bin/conda shell.bash hook`"' >> $(HOME)/.bashrc
	echo conda activate $(ENV) >> $(HOME)/.bashrc
	echo '' >> $(HOME)/.bashrc
	echo export EDITOR=vim >> $(HOME)/.bashrc
	echo export PYTHONDONTWRITEBYTECODE=1 >> $(HOME)/.bashrc

conda: conda-install conda-env conda-setup
	eval `$(HOME)/miniconda3/bin/conda shell.bash hook` && conda env list
	echo Restart your shell to create and activate conda environment "$(ENV)"

## Conda Distribution

conda-index:
	conda index /zdata/projects/shared/conda/geteigen

conda-clean:
	conda clean --all

## Optional Dependencies 

install-mmdet:
	# cd submodules/mmdetection; pip install -e .
	pip install "git+https://github.com/open-mmlab/mmdetection"

## Submoduless

setup-deep-sort:
	cd submodules/deep_sort; \
		git config pull.rebase false; \
		git remote add upstream https://github.com/nwojke/deep_sort

pull-deep-sort:
	cd submodules/deep_sort; \
		git pull; \
		git fetch upstream; \
		git checkout main; \
		git merge upstream/main;

## Local Development 

LOCAL_TORCH_CUDA_ARCH_LIST := $(shell python -c "import torch as th; print('.'.join(map(str, th.cuda.get_device_capability(th.cuda.default_stream().device))))")

dev:
	git config --global credential.helper cache --timeout=21600
	git checkout dev
	make co

dev-setup-local: dev
	TORCH_CUDA_ARCH_LIST=$(LOCAL_TORCH_CUDA_ARCH_LIST) \
		pip install -vv  --force-reinstall --no-deps --no-build-isolation --no-binary :all: -e .

dev-setup: dev
	pip install -vv  --force-reinstall --no-deps --no-build-isolation --no-binary :all: -e .

dev-clean:
	pip uninstall $$(basename -s .git `git config --get remote.origin.url`)
	python setup.py clean --all

## VCS

require-version:
ifndef version
	$(error version is undefined)
endif

clone:
	git clone --recursive $(url) $(dest)

checkout:
	git submodule update --init --recursive
	#cd submodules/mmdetection; git clean -fd; git $@ -f v2.1.0
	branch=$$(git symbolic-ref --short HEAD); \
		echo $$branch; \
		git submodule foreach -q --recursive "git checkout $$(git config -f $$toplevel/.gitmodules submodule.$$name.branch || echo $$branch)"

co: checkout

pull: co
	git submodule update --remote --merge --recursive
	git pull

merge:
	git checkout main
	git merge dev
	git push

tag: require-version
	git checkout main
	git tag -a v$(version) -m v$(version) 
	git push origin tags/v$(version)

del-tag:
	git tag -d $(tag)
	git push origin --delete tags/$(tag)

release:
	git checkout $(git describe --abbrev=0 --tags)
