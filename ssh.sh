#!/bin/bash

open_remote () {
    read -p "Enter an ip to connect to: " ip
    read -p "Enter a port: " port
    
    repo_ref="SeveralLargeRocks/MLXWeek5"

    repo_url="https://github.com/$repo_ref"

    repo_name=$(basename "$repo_url")

    remote_homedir="$(ssh root@${ip} -p ${port} "pwd")"

    ssh root@${ip} -p ${port} -t "git clone ${repo_url}"

    echo "remote_homedir: ${remote_homedir}"

    repo_path="$remote_homedir/$repo_name"

    echo "repo path: ${repo_path}"
    
    echo "opening ${repo_path} on remote"

    code --remote ssh-remote+root@${ip}:${port} $repo_path
}

open_remote
