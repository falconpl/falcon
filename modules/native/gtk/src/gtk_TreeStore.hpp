#ifndef GTK_TREESTORE_HPP
#define GTK_TREESTORE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeStore
 */
class TreeStore
    :
    public Gtk::CoreGObject
{
public:

    TreeStore( const Falcon::CoreClass*, const GtkTreeStore* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_column_types( VMARG );

    static FALCON_FUNC set_value( VMARG );

    static FALCON_FUNC set( VMARG );

#if 0 // unused
    static FALCON_FUNC set_valist( VMARG );
    static FALCON_FUNC set_valuesv( VMARG );
#endif

    static FALCON_FUNC remove( VMARG );

    static FALCON_FUNC insert( VMARG );

    static FALCON_FUNC insert_before( VMARG );

    static FALCON_FUNC insert_after( VMARG );

    static FALCON_FUNC insert_with_values( VMARG );

#if 0 // unused
    static FALCON_FUNC insert_with_valuesv( VMARG );
#endif

    static FALCON_FUNC prepend( VMARG );

    static FALCON_FUNC append( VMARG );

    static FALCON_FUNC is_ancestor( VMARG );

    static FALCON_FUNC iter_depth( VMARG );

    static FALCON_FUNC clear( VMARG );

    static FALCON_FUNC iter_is_valid( VMARG );

    static FALCON_FUNC reorder( VMARG );

    static FALCON_FUNC swap( VMARG );

    static FALCON_FUNC move_before( VMARG );

    static FALCON_FUNC move_after( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TREESTORE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
