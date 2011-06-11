#ifndef GTK_FIXED_HPP
#define GTK_FIXED_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Fixed
 */
class Fixed
    :
    public Gtk::CoreGObject
{
public:

    Fixed( const Falcon::CoreClass*, const GtkFixed* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC put( VMARG );

    static FALCON_FUNC move( VMARG );

    static FALCON_FUNC get_has_window( VMARG );

    static FALCON_FUNC set_has_window( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_FIXED_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
