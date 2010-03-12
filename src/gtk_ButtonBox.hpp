#ifndef GTK_BUTTONBOX_HPP
#define GTK_BUTTONBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::ButtonBox
 */
namespace ButtonBox {

void modInit( Falcon::Module* );

//FALCON_FUNC get_layout( VMARG );

FALCON_FUNC get_child_size( VMARG );

FALCON_FUNC get_child_ipadding( VMARG );

FALCON_FUNC get_child_secondary( VMARG );

//FALCON_FUNC set_spacing( VMARG );

//FALCON_FUNC set_layout( VMARG );

//FALCON_FUNC set_child_size( VMARG );

//FALCON_FUNC set_child_ipadding( VMARG );

FALCON_FUNC set_child_secondary( VMARG );


} // ButtonBox
} // Gtk
} // Falcon

#endif // !GTK_BUTTONBOX_HPP
