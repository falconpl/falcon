########################################################àà
# Falcon CMAKE suite
# Useful functions to finalize modules
#


macro( falcon_define_module varname modname )
      set( ${varname} "${modname}_fm" )
endmacro()


function(falcon_install_module2 tgt dir )
   if(APPLE)
      set_target_properties(${tgt} PROPERTIES
         PREFIX ""
         SUFFIX ".dylib" )
   else()
      set_target_properties(${tgt} PROPERTIES
         PREFIX "" )
   endif()

   set( CMAKE_INSTALL_PREFIX, "@CMAKE_INSTALL_PREFIX@" )

   if( DEFINED MOD_INSTALL )
      set( dest "${MOD_INSTALL}/${dir}" )
   else()
      set( dest "${Falcon_MOD_DIR}/${dir}" )
   endif()

   install( TARGETS ${tgt}
            DESTINATION ${dest} )
endfunction()

function(falcon_install_module tgt )
   falcon_install_module2( "${tgt}" .)
endfunction()

function(falcon_finalize_module2 tgt libs)
      target_link_libraries(${tgt} ${Falcon_LIBRARIES} ${libs} )
      falcon_install_module( ${tgt} )
endfunction()

function(falcon_finalize_module tgt )
      target_link_libraries(${tgt} ${Falcon_LIBRARIES} )
      falcon_install_module( ${tgt} )
endfunction()
