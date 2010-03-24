##
#   Falcon GTK+ module
#

falcon_define_module( CURRENT_MODULE gtk )

##
#   Sources
#
set( SRC_FILES
  g_Object.cpp
  g_ParamSpec.cpp
  gdk_DragContext.cpp
  gdk_EventButton.cpp
  gdk_Pixbuf.cpp
  gtk_Action.cpp
  gtk_Adjustment.cpp
  gtk_Alignment.cpp
  gtk_Arrow.cpp
  gtk_Bin.cpp
  gtk_Box.cpp
  gtk_ButtonBox.cpp
  gtk_Button.cpp
  gtk_CheckButton.cpp
  gtk_Container.cpp
  gtk_enums.cpp
  gtk_Entry.cpp
  gtk_EntryBuffer.cpp
  gtk_EventBox.cpp
  gtk_Fixed.cpp
  gtk_HBox.cpp
  gtk_HButtonBox.cpp
  gtk_HPaned.cpp
  gtk_Image.cpp
  gtk_Label.cpp
  gtk_Layout.cpp
  gtk_Main.cpp
  gtk_Misc.cpp
  gtk_Object.cpp
  gtk_Paned.cpp
  gtk_RadioAction.cpp
  gtk_RadioButton.cpp
  gtk_Requisition.cpp
  gtk_Stock.cpp
  gtk_Table.cpp
  gtk_ToggleAction.cpp
  gtk_ToggleButton.cpp
  gtk_VBox.cpp
  gtk_VButtonBox.cpp
  gtk_VPaned.cpp
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
  g_ParamSpec.hpp
  gdk_DragContext.hpp
  gdk_EventButton.hpp
  gdk_Pixbuf.hpp
  gtk_Action.hpp
  gtk_Adjustment.hpp
  gtk_Alignment.hpp
  gtk_Arrow.hpp
  gtk_Bin.hpp
  gtk_Box.hpp
  gtk_ButtonBox.hpp
  gtk_Button.hpp
  gtk_CheckButton.hpp
  gtk_Container.hpp
  gtk_enums.hpp
  gtk_Entry.hpp
  gtk_EntryBuffer.hpp
  gtk_EventBox.hpp
  gtk_Fixed.hpp
  gtk_HBox.hpp
  gtk_HButtonBox.hpp
  gtk_HPaned.hpp
  gtk_Image.hpp
  gtk_Label.hpp
  gtk_Layout.hpp
  gtk_Main.hpp
  gtk_Misc.hpp
  gtk_Object.hpp
  gtk_Paned.hpp
  gtk_RadioAction.hpp
  gtk_RadioButton.hpp
  gtk_Requisition.hpp
  gtk_Stock.hpp
  gtk_Table.hpp
  gtk_ToggleAction.hpp
  gtk_ToggleButton.hpp
  gtk_VBox.hpp
  gtk_VButtonBox.hpp
  gtk_VPaned.hpp
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
    ${Falcon_LIBRARIES}
    ${GTK2_LIBRARIES}
    gobject-2.0
    )

##
#   Install
#
falcon_install_module( ${CURRENT_MODULE} )
