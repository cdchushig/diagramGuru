# -*- mode: ruby -*-
# vi: set ft=ruby :

API_VERSION = "2"
PROJECT_NAME = File.basename(Dir.pwd)

Vagrant.configure(API_VERSION) do |config|
    config.vm.box = "ubuntu/bionic64"
    config.vm.network "forwarded_port", guest: 80, host: 8080
    config.vm.network "forwarded_port", guest: 443, host: 8443
    config.vm.network "private_network", ip: "192.168.33.10"
    config.vm.network "public_network"

    config.vm.hostname = "vagrant-drscratchv3"
    config.ssh.forward_agent = true
    config.vm.synced_folder "./", "/var/www/drscratchv3"

    config.vm.provision :ansible do |ansible|
        ansible.playbook = "ansible/vagrant.yml"
        ansible.host_key_checking = false
        ansible.verbose = "vv"
    end
end