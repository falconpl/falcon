#ifndef GTK_MENUBAR_HPP
#define GTK_MENUBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::MenuBar
 */
class MenuBar
    :
    public Gtk::CoreGObject
{
public:

    MenuBar( const Falcon::CoreClass*, const GtkMenuBar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if 0 // deprecated
    static FALCON_FUNC append( VMARG );

    static FALCON_FUNC prepend( VMARG );

    static FALCON_FUNC insert( VMARG );
#endif

    static FALCON_FUNC set_pack_direction( VMARG );

    static FALCON_FUNC get_pack_direction( VMARG );

    static FALCON_FUNC set_child_pack_direction( VMARG );

    static FALCON_FUNC get_child_pack_direction( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_MENUBAR_HPP
