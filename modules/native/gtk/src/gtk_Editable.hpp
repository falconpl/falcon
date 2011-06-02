#ifndef GTK_EDITABLE_HPP
#define GTK_EDITABLE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Editable
 */
namespace Editable {

void clsInit( Falcon::Module*, Falcon::Symbol* );

FALCON_FUNC select_region( VMARG );

FALCON_FUNC get_selection_bounds( VMARG );

FALCON_FUNC insert_text( VMARG );

FALCON_FUNC delete_text( VMARG );

FALCON_FUNC get_chars( VMARG );

FALCON_FUNC cut_clipboard( VMARG );

FALCON_FUNC copy_clipboard( VMARG );

FALCON_FUNC paste_clipboard( VMARG );

FALCON_FUNC delete_selection( VMARG );

FALCON_FUNC set_position( VMARG );

FALCON_FUNC get_position( VMARG );

FALCON_FUNC set_editable( VMARG );

FALCON_FUNC get_editable( VMARG );


} // Editable
} // Gtk
} // Falcon

#endif // !GTK_EDITABLE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
