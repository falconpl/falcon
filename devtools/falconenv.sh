#!/bin/sh

if [ -z "$1" ]; then
   echo "falconenv.sh -- Sets the environment for pre-installation falcon tests"
   echo "Usage:"
   echo "   source falconenv.sh <prefix> [libdir] [bindir]"
else

   LIBDIR=$2
   if [ -z "$LIBDIR" ]; then LIBDIR="lib"; fi
   BINDIR=$3
   if [ -z "$BINDIR" ]; then BINDIR="bin"; fi

   export LD_LIBRARY_PATH="$1/$LIBDIR:$LD_LIBRARY_PATH"
   export FALCON_LOAD_PATH=".;$1/$LIBDIR/falcon"
   export PATH="$1/$BINDIR:$PATH"

fi