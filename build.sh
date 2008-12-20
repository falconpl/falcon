#!/bin/sh

func_complete_build()
{
cat << EOT

######################################################
#     BUILD IS SUCCESFULL                            #
#                                                    #
# You can install dbus                     #
# on this system issuing the                         #
#                                                    #
#     $ ./build.sh -i                                #
#                                                    #
# command                                            #
######################################################
EOT
exit 0
}

func_complete_inst()
{
cat << EOT

######################################################
#     INSTALL IS SUCCESFULL                          #
#                                                    #
# Falcon is installed in the target path.            #
# You should run                                     #
#                                                    #
#   $ ldconfig                                       #
#                                                    #
# to update system library cache                     #
######################################################
EOT
exit 0
}

func_errors()
{
cat << EOT
######################################################
#     BUILD PROCESS FAILED!                          #
#                                                    #
# We are sorry, something went wrong. Please, verify #
# the dependencies and other pre-requisite listed    #
# in the README file are correctly set-up.           #
#                                                    #
# In that case, please report the error conditions   #
# to                                                 #
#                                                    #
#       http://www.falconpl.org                      #
#       (Contacts area)                              #
#                                                    #
# Thanks for your cooperation                        #
######################################################
EOT
exit 1
}

func_usage()
{
cat << EOT

Falcon source package build and install tool
Configured for dbus
usage:
         $0 [options]
  -or-   $0 -i [options]
        -i Perform installation step

   Other options
        -j Number of processors to use in make (default 1)
        -d Compile a debug enabled version.
        -s Do NOT strip binaries even if in reselase mode.
        -- Pass the other options to CMAKE

   Environment variables
      CFLAGS - extra C flags to pass to the compiler
      CXXFLAGS - extra C++ flags to pass to the compiler
EOT
}


cat << EOT
######################################################
#     Falcon source distribution build facility      #
######################################################
EOT

TARGET_DEST=""
TARGET_LIB_DIR="lib"
FINAL_DEST=""
DO_INSTALL="no"
DEBUG="no"
STRIP=""
PROCESSORS="1"

until [ -z "$1" ]; do
   case "$1" in
      "-i") DO_INSTALL="yes";;
      "-j") shift; PROCESSORS=$1;;
      "-d") DEBUG="yes";;
      "-s") STRIP="OFF";;
      "--") break;;
      *) func_usage ; exit 0 ;;
   esac
   shift
done


if [ "x$DEBUG" = "xyes" ]; then
   BUILD_NAME="debug"
   STRIP="OFF"
else
   BUILD_NAME="release"
   if [ -z "$STRIP" ]; then
      STRIP="ON"
   fi
fi


if [ "x$DO_INSTALL" = "xyes" ]; then

cat << EOT

      Performing installation

######################################################
EOT
cd "$BUILD_NAME"
make install || func_errors
func_complete_inst

else

cat << EOT

   Configuring and building environment
######################################################

EOT

echo "Launching CMAKE"
mkdir -p "$BUILD_NAME"
cd "$BUILD_NAME"

cmake $* -DCMAKE_CXX_FLAGS:STRING="$CXXFLAGS" \
      -DCMAKE_C_FLAGS:STRING="$CFLAGS" \
      -DCMAKE_BUILD_TYPE:STRING=$BUILD_NAME \
      -DFALCON_STRIP_TARGET:BOOL=$STRIP \
      .. \
   || func_errors

make -j $PROCESSORS || func_errors
func_complete_build
fi

