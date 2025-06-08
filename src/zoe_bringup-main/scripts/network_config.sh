
#Set MTU to 9000
#Modify rmem_max to 33554432 in /etc/sysctl.conf

sudo ethtool -C enp10s0 rx-usecs 10 tx-usecs 10
sudo ethtool -G enp10s0 rx 8184 tx 8184


