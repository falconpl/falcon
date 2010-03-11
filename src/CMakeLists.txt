##
#   Falcon GTK+ module
#

falcon_define_module( CURRENT_MODULE gtk )

##
#   Sources
#
file( GLOB SRC_FILES "modgtk*.cpp" "gtk_*.cpp" )

##
#   Target
#
add_library( ${CURRENT_MODULE} MODULE ${SRC_FILES} )

##
#   Linking
#
target_link_libraries( ${CURRENT_MODULE}
    falcon_engine
    ${GTK2_LIBRARIES}
    )

##
#   Install
#
falcon_install_module( ${CURRENT_MODULE} )
