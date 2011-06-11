# FalconGTK-ARMPokyLinux.cmake

message( STATUS "-- Compiling for ARM-Poky-Linux..." )

set( POKY_ROOT /usr/local/poky/eabi-glibc/arm/arm-poky-linux-gnueabi )

include_directories( BEFORE
    ${POKY_ROOT}/usr/include/gtk-2.0
    ${POKY_ROOT}/usr/lib/gtk-2.0/include
    ${POKY_ROOT}/usr/include/glib-2.0
    ${POKY_ROOT}/usr/lib/glib-2.0/include
    ${POKY_ROOT}/usr/include/pango-1.0
    ${POKY_ROOT}/usr/include/cairo
    ${POKY_ROOT}/usr/include/atk-1.0 )

set( GTK2_LIBRARIES
    ${POKY_ROOT}/usr/lib/libglib-2.0.so
    ${POKY_ROOT}/usr/lib/libgobject-2.0.so
    ${POKY_ROOT}/usr/lib/libgdk-x11-2.0.so
    ${POKY_ROOT}/usr/lib/libgtk-x11-2.0.so
    ${POKY_ROOT}/usr/lib/libcairo.so
    ${POKY_ROOT}/usr/lib/libpango-1.0.so
    ${POKY_ROOT}/usr/lib/libatk-1.0.so )

# vi: set ai et sw=4:
# kate: replace-tabs on; shift-width 4;
