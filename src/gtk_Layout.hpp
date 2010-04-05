#ifndef GTK_LAYOUT_HPP
#define GTK_LAYOUT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Layout
 */
class Layout
    :
    public Gtk::CoreGObject
{
public:

    Layout( const Falcon::CoreClass*, const GtkLayout* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC put( VMARG );

    static FALCON_FUNC move( VMARG );

    static FALCON_FUNC set_size( VMARG );

    static FALCON_FUNC get_size( VMARG );

    static FALCON_FUNC get_hadjustment( VMARG );

    static FALCON_FUNC get_vadjustment( VMARG );

    static FALCON_FUNC set_hadjustment( VMARG );

    static FALCON_FUNC set_vadjustment( VMARG );

    //static FALCON_FUNC get_bin_window( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_LAYOUT_HPP
