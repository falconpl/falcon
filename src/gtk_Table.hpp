#ifndef GTK_TABLE_HPP
#define GTK_TABLE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Table
 */
namespace Table {

void modInit( Falcon::Module* );

FALCON_FUNC init( VMARG );

FALCON_FUNC resize( VMARG );

FALCON_FUNC attach( VMARG );

FALCON_FUNC attach_defaults( VMARG );

FALCON_FUNC set_row_spacing( VMARG );

FALCON_FUNC set_col_spacing( VMARG );

FALCON_FUNC set_row_spacings( VMARG );

FALCON_FUNC set_col_spacings( VMARG );

FALCON_FUNC set_homogeneous( VMARG );

FALCON_FUNC get_default_row_spacing( VMARG );

FALCON_FUNC get_homogeneous( VMARG );

FALCON_FUNC get_row_spacing( VMARG );

FALCON_FUNC get_col_spacing( VMARG );

FALCON_FUNC get_default_col_spacing( VMARG );


} // Table
} // Gtk
} // Falcon

#endif // !GTK_TABLE_HPP
