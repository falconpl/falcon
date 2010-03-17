##
#   Falcon GTK+ module
#

falcon_define_module( CURRENT_MODULE gtk )

##
#   Sources
#
set( SRC_FILES
  g_Object.cpp
  gtk_Arrow.cpp
  gtk_Bin.cpp
  gtk_Box.cpp
  gtk_ButtonBox.cpp
  gtk_Button.cpp
  gtk_CheckButton.cpp
  gtk_Container.cpp
  gtk_enums.cpp
  gtk_HBox.cpp
  gtk_HButtonBox.cpp
  gtk_Label.cpp
  gtk_Main.cpp
  gtk_Misc.cpp
  gtk_Object.cpp
  gtk_RadioButton.cpp
  gtk_Requisition.cpp
  gtk_Table.cpp
  gtk_ToggleButton.cpp
  gtk_VBox.cpp
  gtk_VButtonBox.cpp
  gtk_Widget.cpp
  gtk_Window.cpp
  modgtk.cpp
  modgtk_st.cpp
)

##
#   Header files.  If not added to the target source, they won't show
#   up in Visual Studio.
#
set( HDR_FILES
  g_Object.hpp
  gtk_Arrow.hpp
  gtk_Bin.hpp
  gtk_Box.hpp
  gtk_ButtonBox.hpp
  gtk_Button.hpp
  gtk_CheckButton.hpp
  gtk_Container.hpp
  gtk_enums.hpp
  gtk_HBox.hpp
  gtk_HButtonBox.hpp
  gtk_Label.hpp
  gtk_Main.hpp
  gtk_Misc.hpp
  gtk_Object.hpp
  gtk_RadioButton.hpp
  gtk_Requisition.hpp
  gtk_Table.hpp
  gtk_ToggleButton.hpp
  gtk_VBox.hpp
  gtk_VButtonBox.hpp
  gtk_Widget.hpp
  gtk_Window.hpp
  modgtk.hpp
  modgtk_st.hpp
  modgtk_version.hpp
)

##
#   Target
#
add_library( ${CURRENT_MODULE} MODULE
  ${SRC_FILES}
  ${HDR_FILES}
)

##
#   Linking
#
target_link_libraries( ${CURRENT_MODULE}
    falcon_engine
    ${GTK2_LIBRARIES}
    gobject-2.0
    )

##
#   Install
#
falcon_install_module( ${CURRENT_MODULE} )
