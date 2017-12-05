# # In case of locale error
# export LANGUAGE=en_US.UTF-8
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8

# General dependencies
sudo apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    dist-upgrade
sudo apt-get install -y build-essential g++ gcc gfortran wget git \
    linux-image-generic linux-headers-generic libopenblas-dev htop \
    libfreetype6-dev libxft-dev libncurses-dev libblas-dev \
    liblapack-dev libatlas-base-dev linux-image-extra-virtual unzip \
    swig pkg-config zip zlib1g-dev libcurl3-dev

# # Assume the machine has nvidia graphics driver pre-installed (similar to AWS p2.xlarge), if not run the following
# sudo add-apt-repository -y ppa:graphics-drivers/ppa
# sudo apt-get update
# sudo apt-get install -y nvidia-375
# # check installation
# nvidia-smi
# # might need to restart machine after installation in case nvidia-smi failure
# sudo reboot

# CUDA 8.0
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" install cuda
rm -f cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export CUDA_ROOT=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_ROOT/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64' >> ~/.bashrc
source ~/.bashrc

# OpenCV with CUDA acceleration
sudo apt-get install -y liblapacke-dev checkinstall
git clone https://github.com/Jim61C/opencv_install_sh.git
cd opencv_install_sh
chmod 755 install_opencv.sh
./install_opencv.sh
cd ..

# caffe-fast-rcnn, download caffe-fast-rcnn source, branch faster-rcnn-upstream-33f2445
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends -y libboost-all-dev
sudo apt-get install -y libopenblas-dev
sudo apt-get install -y libgflags-dev libgtest-dev libc++-dev clang
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

git clone https://github.com/Jim61C/caffe-fast-rcnn.git
cd caffe-fast-rcnn
git fetch -a
git checkout faster-rcnn-upstream-33f2445-custom-compilation
make all
make test
make runtest
make pycaffe
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
cd ..

#TinyXML
sudo apt-get install -y libtinyxml-dev

# Boost
sudo apt-get install -y libboost-all-dev

# Eigen
sudo apt-get install -y libeigen3-dev

# GNU Scientific Library (GSL)
sudo apt install -y libgsl2 libgsl-dev

# TRAX
sudo add-apt-repository -y ppa:lukacu/trax
sudo apt-get update
sudo apt-get install -y libtrax-opencv0 libtrax-opencv-dev