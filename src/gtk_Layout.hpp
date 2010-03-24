#ifndef GTK_LAYOUT_HPP
#define GTK_LAYOUT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Layout
 */
namespace Layout {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC put( VMARG );

FALCON_FUNC move( VMARG );

FALCON_FUNC set_size( VMARG );

FALCON_FUNC get_size( VMARG );

FALCON_FUNC get_hadjustment( VMARG );

FALCON_FUNC get_vadjustment( VMARG );

FALCON_FUNC set_hadjustment( VMARG );

FALCON_FUNC set_vadjustment( VMARG );

//FALCON_FUNC get_bin_window( VMARG );


} // Layout
} // Gtk
} // Falcon

#endif // !GTK_LAYOUT_HPP
