#ifndef GTK_FRAME_HPP
#define GTK_FRAME_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Frame
 */
namespace Frame {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC set_label( VMARG );

FALCON_FUNC set_label_widget( VMARG );

FALCON_FUNC set_label_align( VMARG );

FALCON_FUNC set_shadow_type( VMARG );

FALCON_FUNC get_label( VMARG );

FALCON_FUNC get_label_align( VMARG );

FALCON_FUNC get_label_widget( VMARG );

FALCON_FUNC get_shadow_type( VMARG );


} // Frame
} // Gtk
} // Falcon

#endif // !GTK_FRAME_HPP
