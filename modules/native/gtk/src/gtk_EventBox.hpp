#ifndef GTK_EVENTBOX_HPP
#define GTK_EVENTBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::EventBox
 */
class EventBox
    :
    public Gtk::CoreGObject
{
public:

    EventBox( const Falcon::CoreClass*, const GtkEventBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_above_child( VMARG );

    static FALCON_FUNC get_above_child( VMARG );

    static FALCON_FUNC set_visible_window( VMARG );

    static FALCON_FUNC get_visible_window( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_EVENTBOX_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
