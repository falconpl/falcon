#ifndef GTK_TOOLITEMGROUP_HPP
#define GTK_TOOLITEMGROUP_HPP

#include "modgtk.hpp"

#if GTK_CHECK_VERSION( 2, 20, 0 )

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ToolItemGroup
 */
class ToolItemGroup
    :
    public Gtk::CoreGObject
{
public:

    ToolItemGroup( const Falcon::CoreClass*, const GtkToolItemGroup* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC get_collapsed( VMARG );

    static FALCON_FUNC get_drop_item( VMARG );

    static FALCON_FUNC get_ellipsize( VMARG );

    static FALCON_FUNC get_item_position( VMARG );

    static FALCON_FUNC get_n_items( VMARG );

    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC get_label_widget( VMARG );

    static FALCON_FUNC get_nth_item( VMARG );

    static FALCON_FUNC get_header_relief( VMARG );

    static FALCON_FUNC insert( VMARG );

    static FALCON_FUNC set_collapsed( VMARG );

    static FALCON_FUNC set_ellipsize( VMARG );

    static FALCON_FUNC set_item_position( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC set_label_widget( VMARG );

    static FALCON_FUNC set_header_relief( VMARG );

};


} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 20, 0 )

#endif // !GTK_TOOLITEMGROUP_HPP
