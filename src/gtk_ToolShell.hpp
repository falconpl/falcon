#ifndef GTK_TOOLSHELL_HPP
#define GTK_TOOLSHELL_HPP

#include "modgtk.hpp"


#if GTK_MINOR_VERSION >= 14

namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::ToolShell
 */
namespace ToolShell {

void clsInit( Falcon::Module*, Falcon::Symbol* );

#if GTK_MINOR_VERSION >= 20
FALCON_FUNC get_ellipsize_mode( VMARG );
#endif

FALCON_FUNC get_icon_size( VMARG );

FALCON_FUNC get_orientation( VMARG );

FALCON_FUNC get_relief_style( VMARG );

FALCON_FUNC get_style( VMARG );

#if GTK_MINOR_VERSION >= 20
FALCON_FUNC get_text_alignment( VMARG );

FALCON_FUNC get_text_orientation( VMARG );
#endif

FALCON_FUNC rebuild_menu( VMARG );

#if GTK_MINOR_VERSION >= 20
//FALCON_FUNC get_text_size_group( VMARG );
#endif


} // ToolShell
} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 14

#endif // !GTK_TOOLSHELL_HPP
