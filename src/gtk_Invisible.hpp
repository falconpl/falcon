#ifndef GTK_INIVISIBLE_HPP
#define GTK_INIVISIBLE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Invisible
 */
class Invisible
    :
    public Gtk::CoreGObject
{
public:

    Invisible( const Falcon::CoreClass*, const GtkInvisible* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );
#if 0
    static FALCON_FUNC new_for_screen( VMARG );

    static FALCON_FUNC set_screen( VMARG );

    static FALCON_FUNC get_screen( VMARG );
#endif
};


} // Gtk
} // Falcon

#endif // !GTK_INIVISIBLE_HPP
