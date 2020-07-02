# If not running interactively, do nothing
if [ -z "$PS1" ] ; then
  return 
fi

export PS1="\[\033[01;32m\]adapt-docker@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"

if [ $EUID -eq 0 ] ; then
  echo -e "\e[0;33m"
  cat << WARN
WARNING: The user inside container is root, which might cause new files in mounted
volumes to be created as the root user on your host machine.

To avoid this, run the container again with the arguments that specifies the userid
of the user inside the container as your userid on your host machine:

$ docker run -u \$(id -u):\$(id -g) args ...
WARN
  echo -e "\e[m"
fi
