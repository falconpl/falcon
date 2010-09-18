#ifndef GTK_TABLE_HPP
#define GTK_TABLE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Table
 */
class Table
    :
    public Gtk::CoreGObject
{
public:

    Table( const Falcon::CoreClass*, const GtkTable* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC resize( VMARG );

    static FALCON_FUNC attach( VMARG );

    static FALCON_FUNC attach_defaults( VMARG );

    static FALCON_FUNC set_row_spacing( VMARG );

    static FALCON_FUNC set_col_spacing( VMARG );

    static FALCON_FUNC set_row_spacings( VMARG );

    static FALCON_FUNC set_col_spacings( VMARG );

    static FALCON_FUNC set_homogeneous( VMARG );

    static FALCON_FUNC get_default_row_spacing( VMARG );

    static FALCON_FUNC get_homogeneous( VMARG );

    static FALCON_FUNC get_row_spacing( VMARG );

    static FALCON_FUNC get_col_spacing( VMARG );

    static FALCON_FUNC get_default_col_spacing( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TABLE_HPP
