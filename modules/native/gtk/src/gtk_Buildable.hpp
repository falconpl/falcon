#ifndef GTK_BUILDABLE_HPP
#define GTK_BUILDABLE_HPP

#include "modgtk.hpp"


#if GTK_CHECK_VERSION( 2, 12, 0 )

namespace Falcon {
namespace Gtk {

/**
 *  \namespace Falcon::Gtk::Buildable
 */
namespace Buildable {

void clsInit( Falcon::Module*, Falcon::Symbol* );

FALCON_FUNC set_name( VMARG );

FALCON_FUNC get_name( VMARG );
#if 0
FALCON_FUNC add_child( VMARG );

FALCON_FUNC set_buildable_property( VMARG );

FALCON_FUNC construct_child( VMARG );

FALCON_FUNC custom_tag_start( VMARG );

FALCON_FUNC custom_tag_end( VMARG );

FALCON_FUNC custom_finished( VMARG );

FALCON_FUNC parser_finished( VMARG );

FALCON_FUNC get_internal_child( VMARG );
#endif

} // Buildable
} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 12, 0 )

#endif // !GTK_BUILDABLE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
