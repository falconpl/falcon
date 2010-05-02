########################################################àà
# Falcon CMAKE suite
# Useful functions to finalize modules
#

macro( falcon_define_module varname modname )
      set( ${varname} "${modname}_fm" )
endmacro( falcon_define_module )

function(FALCON_FINALIZE_MODULE tgt )
      target_link_libraries(${tgt} ${Falcon_LIBRARIES} )
      falcon_install_module( ${tgt} )
endfunction(FALCON_FINALIZE_MODULE)

function(FALCON_FINALIZE_MODULE2 tgt libs)
      target_link_libraries(${tgt} ${Falcon_LIBRARIES} ${libs} )
      falcon_install_module( ${tgt} )
endfunction(FALCON_FINALIZE_MODULE2)

function(FALCON_INSTALL_MODULE tgt )
   if(APPLE)
      set_target_properties(${tgt} PROPERTIES
         PREFIX ""
         SUFFIX ".dylib" )
   else()
      set_target_properties(${tgt} PROPERTIES
         PREFIX "" )
   endif()

   install( TARGETS ${tgt}
            DESTINATION @FALCON_MOD_DIR@ )
endfunction(FALCON_INSTALL_MODULE)
